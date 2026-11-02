"""Train vanilla ARS or ordinary ES on MMK12.

ARS evaluates both sides of 16 isotropic directions and updates the top 8.
ES evaluates 30 one-sided directions and updates with the z-scored rewards.
The official MMK12 test split is not read.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mmk12_zo.data import load_splits
from mmk12_zo.fleet import Fleet
from mmk12_zo.search import es_weights, select_directions
from mmk12_zo.seeds import batch_seed, direction_seed

ARS_DIRECTIONS = 16
ARS_TOP_K = 8
ARS_NOISE = 0.001
ARS_LR = 0.0001
ES_POPULATION = 30
ES_SIGMA = 0.001
ES_ALPHA = 5e-4


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(row[key] for row in rows) / len(rows)


def _sample(train_ids: list[str], step: int, size: int) -> list[str]:
    rng = np.random.Generator(np.random.PCG64(batch_seed(step)))
    picked = rng.choice(np.asarray(train_ids), size=size, replace=False)
    return [str(item) for item in picked]


def _write(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _empty_task(max_tokens: int) -> dict:
    return {
        "op": "generate",
        "ids": [],
        "direction_seed": 0,
        "sign": 0,
        "noise_std": 0.0,
        "max_tokens": max_tokens,
    }


def _run_jobs(fleet: Fleet, jobs: list[dict], ids: list[str], noise_std: float, max_tokens: int) -> dict:
    buckets: list[list[dict]] = [[] for _ in range(fleet.n_gpus)]
    for position, job in enumerate(jobs):
        buckets[position % fleet.n_gpus].append(job)
    waves = max(len(bucket) for bucket in buckets)
    collected = {}
    started = time.perf_counter()
    for wave in range(waves):
        tasks = []
        wave_jobs = []
        for gpu in range(fleet.n_gpus):
            if wave >= len(buckets[gpu]):
                tasks.append(_empty_task(max_tokens))
                wave_jobs.append(None)
                continue
            job = buckets[gpu][wave]
            wave_jobs.append(job)
            tasks.append(
                {
                    "op": "generate",
                    "ids": ids,
                    "direction_seed": job["seed"],
                    "sign": job["sign"],
                    "noise_std": noise_std,
                    "max_tokens": max_tokens,
                }
            )
        replies = fleet.call_each(tasks)
        for job, reply in zip(wave_jobs, replies):
            if job is not None:
                collected[(job["index"], job["sign"])] = reply
    return {"collected": collected, "seconds": time.perf_counter() - started}


def _dev(fleet: Fleet, dev_ids: list[str], step: int, run_dir: str, max_tokens: int) -> dict:
    tasks = []
    for gpu in range(fleet.n_gpus):
        tasks.append(
            {
                "op": "generate",
                "ids": dev_ids[gpu :: fleet.n_gpus],
                "direction_seed": 0,
                "sign": 0,
                "noise_std": 0.0,
                "max_tokens": max_tokens,
            }
        )
    started = time.perf_counter()
    replies = fleet.call_each(tasks)
    rows = [row for reply in replies for row in reply["rows"]]
    rows.sort(key=lambda row: row["id"])
    path = os.path.join(run_dir, "dev", f"step_{step:04d}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    correct = sum(row["reward"] for row in rows)
    summary = {
        "step": step,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "format_rate": _mean(rows, "format_ok"),
        "seconds": time.perf_counter() - started,
    }
    _write(os.path.join(run_dir, "dev_curve.jsonl"), summary)
    print(
        f"dev step {step}: {correct}/{len(rows)} format {summary['format_rate']:.3f} in {summary['seconds']:.1f}s",
        flush=True,
    )
    return summary


def _save_and_sync(fleet: Fleet, run_dir: str, step: int, directions: list, scale: float) -> dict:
    updated = fleet.call_one(0, {"op": "update", "directions": directions, "scale": scale})
    path = os.path.join(run_dir, "checkpoints", f"step_{step:04d}.safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fleet.call_one(0, {"op": "save", "path": path})
    loaded = fleet.call_all({"op": "load", "path": path})
    checksums = [item["report"]["checksum"] for item in loaded]
    if any(item != checksums[0] for item in checksums):
        raise RuntimeError(f"GPU weights diverged after reload: {checksums}")
    return updated["report"]


def _train_ars(fleet: Fleet, train_ids: list[str], step: int, run_dir: str, batch_size: int, max_tokens: int) -> dict:
    ids = _sample(train_ids, step, batch_size)
    jobs = []
    for index in range(ARS_DIRECTIONS):
        seed = direction_seed(step, index)
        jobs.append({"index": index, "sign": 1, "seed": seed})
        jobs.append({"index": index, "sign": -1, "seed": seed})
    ran = _run_jobs(fleet, jobs, ids, ARS_NOISE, max_tokens)
    rewards_plus = []
    rewards_minus = []
    for index in range(ARS_DIRECTIONS):
        rewards_plus.append(_mean(ran["collected"][(index, 1)]["rows"], "reward"))
        rewards_minus.append(_mean(ran["collected"][(index, -1)]["rows"], "reward"))
    decision = select_directions(rewards_plus, rewards_minus, ARS_TOP_K)
    if decision["skip"]:
        scale = 0.0
        directions = []
    else:
        scale = ARS_LR / (len(decision["selected"]) * decision["sigma"])
        directions = [
            [direction_seed(step, index), diff]
            for index, diff in zip(decision["selected"], decision["diffs"])
        ]
    report = _save_and_sync(fleet, run_dir, step, directions, scale)
    summary = {
        "step": step,
        "method": "ars",
        "question_ids": ids,
        "rewards_plus": rewards_plus,
        "rewards_minus": rewards_minus,
        "selected": decision["selected"],
        "sigma": decision["sigma"],
        "skip": decision["skip"],
        "skip_reason": decision["skip_reason"],
        "update_l2": report.get("l2"),
        "seconds": ran["seconds"],
    }
    _write(os.path.join(run_dir, "steps.jsonl"), summary)
    print(
        f"ars step {step}: mean {(sum(rewards_plus) + sum(rewards_minus)) / (2 * ARS_DIRECTIONS):.3f} "
        f"selected {len(decision['selected'])} skip {decision['skip']} {decision['skip_reason']} "
        f"l2 {report.get('l2')} wall {ran['seconds']:.1f}s",
        flush=True,
    )
    return summary


def _train_es(fleet: Fleet, train_ids: list[str], step: int, run_dir: str, batch_size: int, max_tokens: int) -> dict:
    ids = _sample(train_ids, step, batch_size)
    jobs = [{"index": index, "sign": 1, "seed": direction_seed(step, index)} for index in range(ES_POPULATION)]
    ran = _run_jobs(fleet, jobs, ids, ES_SIGMA, max_tokens)
    rewards = [_mean(ran["collected"][(index, 1)]["rows"], "reward") for index in range(ES_POPULATION)]
    decision = es_weights(rewards)
    if decision["skip"]:
        scale = 0.0
        directions = []
    else:
        scale = ES_ALPHA / ES_POPULATION
        directions = [
            [direction_seed(step, index), weight] for index, weight in enumerate(decision["weights"])
        ]
    report = _save_and_sync(fleet, run_dir, step, directions, scale)
    summary = {
        "step": step,
        "method": "es",
        "question_ids": ids,
        "rewards": rewards,
        "sigma": decision["sigma"],
        "skip": decision["skip"],
        "skip_reason": decision["skip_reason"],
        "update_l2": report.get("l2"),
        "seconds": ran["seconds"],
    }
    _write(os.path.join(run_dir, "steps.jsonl"), summary)
    print(
        f"es step {step}: mean {sum(rewards) / len(rewards):.3f} sigma {decision['sigma']:.4f} "
        f"skip {decision['skip']} {decision['skip_reason']} l2 {report.get('l2')} wall {ran['seconds']:.1f}s",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ars", "es"), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("MMK12_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    args = parser.parse_args()
    if args.method == "ars" and (ARS_DIRECTIONS * 2) % args.gpus != 0:
        raise SystemExit("ARS has 32 candidates; the GPU count must divide 32")
    os.environ["MMK12_MODEL"] = args.model
    splits = load_splits()
    train_ids = list(splits["train_ids"])
    dev_ids = list(splits["dev_ids"])
    os.makedirs(args.run_dir, exist_ok=True)
    config = {
        "method": args.method,
        "model": args.model,
        "dataset": "FanqingM/MMK12",
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "steps": args.steps,
        "gpus": args.gpus,
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_test_held_out": len(splits["test_ids_held_out"]),
        "ars": {
            "directions": ARS_DIRECTIONS,
            "top_k": ARS_TOP_K,
            "noise_std": ARS_NOISE,
            "learning_rate": ARS_LR,
        },
        "es": {"population": ES_POPULATION, "sigma": ES_SIGMA, "alpha": ES_ALPHA},
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42},
    }
    with open(os.path.join(args.run_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    state_path = os.path.join(args.run_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"completed_step": -1}
    fleet = Fleet(args.gpus)
    try:
        fleet.start()
        completed = int(state["completed_step"])
        center = args.model if completed < 0 else os.path.join(args.run_dir, "checkpoints", f"step_{completed:04d}.safetensors")
        loaded = fleet.call_all({"op": "load", "path": center})
        print("loaded", loaded[0]["report"], flush=True)
        if completed < 0:
            dev = _dev(fleet, dev_ids, 0, args.run_dir, args.max_tokens)
            state = {"completed_step": 0, "dev0": dev}
            with open(state_path, "w") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
            completed = 0
        train = _train_ars if args.method == "ars" else _train_es
        for step in range(completed + 1, args.steps + 1):
            train(fleet, train_ids, step, args.run_dir, args.batch_size, args.max_tokens)
            state = {"completed_step": step}
            if step % 5 == 0 or step == args.steps:
                dev = _dev(fleet, dev_ids, step, args.run_dir, args.max_tokens)
                state["last_dev"] = dev
            with open(state_path, "w") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
            print(f"completed step {step}", flush=True)
    finally:
        fleet.stop()


if __name__ == "__main__":
    main()
