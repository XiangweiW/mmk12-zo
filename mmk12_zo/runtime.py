"""One vLLM engine. Imported only after CUDA_VISIBLE_DEVICES is set."""

from __future__ import annotations

import inspect
import os
import time
import traceback

from mmk12_zo.data import load_train_and_dev
from mmk12_zo.scoring import grade

SYSTEM_PROMPT = (
    "You are solving a problem shown in the image. "
    "Read the image and the question. "
    "On the last line, put only the final answer in \\boxed{}."
)

_RECORDS = None


def _records():
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = load_train_and_dev()
    return _RECORDS


def _engine_kwargs(model: str) -> dict:
    from vllm import LLM

    kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(os.environ.get("MMK12_GPU_MEMORY_UTILIZATION", "0.90")),
        "max_model_len": int(os.environ.get("MMK12_MAX_MODEL_LEN", "6144")),
        "max_num_seqs": int(os.environ.get("MMK12_MAX_NUM_SEQS", "32")),
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 1},
        "seed": 42,
        "trust_remote_code": False,
        "enforce_eager": False,
        "worker_extension_cls": "mmk12_zo.worker.ZOExtension",
        "mm_processor_cache_gb": 0,
        "generation_config": "vllm",
        "override_generation_config": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        "allowed_local_media_path": os.path.abspath(os.environ.get("MMK12_ROOT", os.getcwd())),
    }
    available = inspect.signature(LLM.__init__).parameters
    has_var_keyword = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in available.values())
    if not has_var_keyword:
        for key in [key for key in list(kwargs) if key not in available]:
            kwargs.pop(key)
    return kwargs


def _conversation(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file://" + os.path.abspath(record["image_path"])}},
                {"type": "text", "text": record["question"]},
            ],
        },
    ]


def _sampling(max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, seed=42)


def serve(gpu_id: int, in_q, out_q) -> None:
    from vllm import LLM

    model = os.environ.get("MMK12_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    llm = LLM(**_engine_kwargs(model))
    out_q.put({"ok": True, "op": "ready", "gpu": gpu_id})
    while True:
        task = in_q.get()
        op = task["op"]
        try:
            if op == "stop":
                out_q.put({"ok": True, "op": "stop", "gpu": gpu_id})
                break
            if op == "load":
                report = llm.collective_rpc("load_weights_from", args=(task["path"],))[0]
                out_q.put({"ok": True, "op": op, "gpu": gpu_id, "report": report})
            elif op == "generate":
                weight = llm.collective_rpc(
                    "set_weights",
                    kwargs={
                        "direction_seed": int(task["direction_seed"]),
                        "sign": int(task["sign"]),
                        "noise_std": float(task["noise_std"]),
                    },
                )[0]
                if hasattr(llm, "reset_prefix_cache"):
                    llm.reset_prefix_cache()
                ids = task["ids"]
                if not ids:
                    out_q.put(
                        {
                            "ok": True,
                            "op": op,
                            "gpu": gpu_id,
                            "seconds": 0.0,
                            "rows": [],
                            "weight": weight,
                        }
                    )
                    continue
                records = _records()
                chats = [_conversation(records[item_id]) for item_id in ids]
                started = time.perf_counter()
                outputs = llm.chat(chats, sampling_params=_sampling(int(task["max_tokens"])), use_tqdm=False)
                seconds = time.perf_counter() - started
                rows = []
                for item_id, output in zip(ids, outputs):
                    text = output.outputs[0].text
                    graded = grade(text, records[item_id]["answer"])
                    rows.append(
                        {
                            "id": item_id,
                            "reward": graded.reward,
                            "format_ok": graded.format_ok,
                            "prediction": graded.prediction,
                            "answer": records[item_id]["answer"],
                            "completion_tokens": len(output.outputs[0].token_ids),
                            "text": text,
                        }
                    )
                out_q.put(
                    {
                        "ok": True,
                        "op": op,
                        "gpu": gpu_id,
                        "seconds": seconds,
                        "rows": rows,
                        "weight": weight,
                    }
                )
            elif op == "update":
                report = llm.collective_rpc(
                    "apply_update",
                    kwargs={"directions": task["directions"], "scale": float(task["scale"])},
                )[0]
                if hasattr(llm, "reset_prefix_cache"):
                    llm.reset_prefix_cache()
                out_q.put({"ok": True, "op": op, "gpu": gpu_id, "report": report})
            elif op == "save":
                report = llm.collective_rpc("save_weights", args=(task["path"],))[0]
                out_q.put({"ok": True, "op": op, "gpu": gpu_id, "report": report})
            else:
                raise ValueError(f"unknown op {op}")
        except Exception:
            out_q.put({"ok": False, "op": op, "gpu": gpu_id, "trace": traceback.format_exc()})
