"""One vLLM engine per GPU. This module does not import torch or vLLM."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time


def _prepare_env(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        path = os.environ.get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def gpu_worker(gpu_id: int, in_q, out_q) -> None:
    _prepare_env(gpu_id)
    from mmk12_zo.runtime import serve

    serve(gpu_id, in_q, out_q)


class Fleet:
    def __init__(self, n_gpus: int):
        self.n_gpus = n_gpus
        self.ctx = mp.get_context("spawn")
        self.in_queues = []
        self.out_q = None
        self.procs = []

    def start(self) -> list[dict]:
        self.out_q = self.ctx.Queue()
        try:
            for gpu_id in range(self.n_gpus):
                in_q = self.ctx.Queue()
                proc = self.ctx.Process(target=gpu_worker, args=(gpu_id, in_q, self.out_q), name=f"zo-gpu{gpu_id}")
                proc.start()
                self.in_queues.append(in_q)
                self.procs.append(proc)
            return [self._recv() for _ in range(self.n_gpus)]
        except Exception:
            self.kill()
            raise

    def kill(self) -> None:
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()

    def stop(self) -> None:
        for in_q, proc in zip(self.in_queues, self.procs):
            if proc.is_alive():
                try:
                    in_q.put({"op": "stop"})
                except Exception:
                    pass
        deadline = time.time() + 30
        while time.time() < deadline and any(proc.is_alive() for proc in self.procs):
            try:
                self.out_q.get(timeout=1)
            except queue.Empty:
                continue
        self.kill()
        for proc in self.procs:
            proc.join(timeout=5)

    def call_all(self, task: dict) -> list[dict]:
        for in_q in self.in_queues:
            in_q.put(task)
        replies = [self._recv() for _ in range(self.n_gpus)]
        replies.sort(key=lambda item: item["gpu"])
        return replies

    def call_one(self, gpu: int, task: dict) -> dict:
        self.in_queues[gpu].put(task)
        reply = self._recv()
        if reply["gpu"] != gpu:
            raise RuntimeError(f"expected GPU {gpu}, got {reply['gpu']}")
        return reply

    def call_each(self, tasks: list[dict]) -> list[dict]:
        if len(tasks) != self.n_gpus:
            raise ValueError("need one task per GPU")
        for in_q, task in zip(self.in_queues, tasks):
            in_q.put(task)
        replies = [self._recv() for _ in range(self.n_gpus)]
        replies.sort(key=lambda item: item["gpu"])
        return replies

    def _recv(self) -> dict:
        reply = self.out_q.get()
        if not reply.get("ok", False):
            raise RuntimeError(f"GPU {reply.get('gpu')} failed during {reply.get('op')}:\n{reply.get('trace')}")
        return reply
