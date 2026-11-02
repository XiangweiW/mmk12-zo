"""Download FanqingM/MMK12 and freeze a train/dev split.

The official test split is saved and never used for training or model selection.
Dev is a fixed 200-id sample of the official train split, seed 42.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mmk12_zo.seeds import mix_seed

DATASET_ID = "FanqingM/MMK12"
DEV_SIZE = 200
SPLIT_SEED = mix_seed(42, 9)


def project_root() -> str:
    return os.environ.get("MMK12_ROOT", os.getcwd())


def data_root() -> str:
    return os.path.join(project_root(), "data", "mmk12")


def splits_path() -> str:
    return os.path.join(data_root(), "splits.json")


def image_path(split: str, item_id: str) -> str:
    return os.path.join(data_root(), "images", split, f"{item_id}.png")


def records_path(split: str) -> str:
    return os.path.join(data_root(), "records", f"{split}.jsonl")


def prepare() -> dict:
    from datasets import load_dataset

    root = data_root()
    os.makedirs(root, exist_ok=True)
    dataset = load_dataset(DATASET_ID)
    counts = {}
    for split in ("train", "test"):
        records = []
        for row in dataset[split]:
            item_id = str(row["id"])
            path = image_path(split, item_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(path)
            records.append(
                {
                    "id": item_id,
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                    "subject": str(row["subject"]),
                    "image_path": path,
                }
            )
        os.makedirs(os.path.dirname(records_path(split)), exist_ok=True)
        with open(records_path(split), "w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(records)
    train_ids = [json.loads(line)["id"] for line in open(records_path("train"))]
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    dev_ids = sorted(rng.choice(np.asarray(train_ids), size=DEV_SIZE, replace=False).tolist())
    dev_set = set(dev_ids)
    pool_ids = [item for item in train_ids if item not in dev_set]
    test_ids = [json.loads(line)["id"] for line in open(records_path("test"))]
    payload = {
        "dataset": DATASET_ID,
        "dev_size": DEV_SIZE,
        "split_seed": SPLIT_SEED,
        "train_ids": pool_ids,
        "dev_ids": dev_ids,
        "test_ids_held_out": test_ids,
        "counts": {"train_pool": len(pool_ids), "dev": len(dev_ids), "test": counts["test"]},
        "note": "test_ids_held_out must not be used for training or checkpoint selection",
    }
    with open(splits_path(), "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"prepared {DATASET_ID}: train {len(pool_ids)} dev {len(dev_ids)} test {counts['test']}",
        flush=True,
    )
    return payload


def load_splits() -> dict:
    with open(splits_path()) as handle:
        return json.load(handle)


def load_records(split: str) -> dict[str, dict]:
    found = {}
    with open(records_path(split)) as handle:
        for line in handle:
            record = json.loads(line)
            found[record["id"]] = record
    return found


def load_train_and_dev() -> dict[str, dict]:
    """Train-pool and dev records. The official test file is not read."""
    splits = load_splits()
    train = load_records("train")
    allowed = set(splits["train_ids"]) | set(splits["dev_ids"])
    return {item_id: train[item_id] for item_id in allowed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    prepare()


if __name__ == "__main__":
    main()
