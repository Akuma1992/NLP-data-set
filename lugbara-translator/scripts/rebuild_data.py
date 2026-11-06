"""Rebuild the SALT data used on Kaggle, on this PC.

* data/salt_sample/  train.jsonl, val.jsonl, test.jsonl, stats.json: the thesis-sized split
  (15,524 sentences, seed 13, 70/15/15) made by the same code as the Kaggle runs
* data/salt_full/    every row of SALT `text-all` (train/dev/test), as JSONL and CSV

Needs this PC signed in to Hugging Face with an account that accepted the SALT terms.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "_rebuild"
os.environ["LUGBARA_WORK"] = str(WORK)
sys.path.insert(0, str(ROOT / "kaggle"))
import lugbara_mt as L  # noqa: E402  (reads LUGBARA_WORK at import)


def main():
    out_sample = ROOT / "data" / "salt_sample"
    out_full = ROOT / "data" / "salt_full"
    out_sample.mkdir(parents=True, exist_ok=True)
    out_full.mkdir(parents=True, exist_ok=True)

    # 1. the split, exactly as on Kaggle
    L.prepare(argparse.Namespace(fake=0))
    for f in ("train.jsonl", "val.jsonl", "test.jsonl", "stats.json"):
        shutil.copy(L.DATA / f, out_sample / f)

    # 2. the whole SALT text table, untouched
    from datasets import load_dataset
    ds = load_dataset(L.SALT, "text-all")
    for split in ds:
        rows = ds[split].to_list()
        with open(out_full / f"text-all.{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_full / f"text-all.{split}.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(split, len(rows), "rows ->", out_full / f"text-all.{split}.jsonl")
    (out_full / "README.txt").write_text(
        "Sunbird SALT, config text-all (https://huggingface.co/datasets/Sunbird/salt), CC BY-SA 4.0.\n"
        "Downloaded with the cakuma account, which accepted the dataset's access terms.\n"
        "Do not re-publish without keeping the licence and attribution.\n", encoding="utf-8")
    shutil.rmtree(WORK, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
