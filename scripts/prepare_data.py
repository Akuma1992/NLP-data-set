"""Build train/dev/test files from Sunbird SALT (text-all) for every direction in DIRECTIONS.

Train = all English<->Lugbara pairs, most Luganda->English pairs, and samples of the
directions the Rutooro model already had, so they are kept while the new ones are learned.
"""
import json
import random

from datasets import load_dataset

from common import COLS, DATA, DIRECTIONS, SALT_REPO, dname, normalize, tag

SEED = 13


def pairs(split, src, tgt):
    out = []
    for r in split:
        s, t = normalize(r[COLS[src]]), normalize(r[COLS[tgt]])
        if s and t:
            out.append({"id": r["id"], "src": s, "tgt": t})
    return out


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    DATA.mkdir(exist_ok=True)
    ds = load_dataset(SALT_REPO, "text-all")
    print({k: len(v) for k, v in ds.items()})

    train, stats = [], {"directions": {}}
    for src, tgt, n in DIRECTIONS:
        d = dname(src, tgt)
        tr = pairs(ds["train"], src, tgt)
        random.shuffle(tr)
        if n:
            tr = tr[:n]
        train += [{"dir": d, "src": f"{tag(tgt)} {it['src']}", "tgt": it["tgt"]} for it in tr]
        dev, test = pairs(ds["dev"], src, tgt), pairs(ds["test"], src, tgt)
        write(DATA / f"dev.{d}.jsonl", dev)
        write(DATA / f"test.{d}.jsonl", test)
        stats["directions"][d] = {"train": len(tr), "dev": len(dev), "test": len(test)}

    # how much of each SALT table actually has text for each language
    stats["coverage"] = {
        split: {"rows": len(ds[split]),
                **{lang: sum(1 for r in ds[split] if normalize(r[col])) for lang, col in COLS.items()}}
        for split in ("train", "dev", "test")}

    # whole corpus (all splits) with the Lugbara side kept raw, for the language analysis
    allrows = []
    for split in ("train", "dev", "test"):
        for r in ds[split]:
            if normalize(r["lgg_text"]):
                allrows.append({"split": split, "id": r["id"], "raw": r["lgg_text"],
                                **{lang: normalize(r[col]) for lang, col in COLS.items()}})
    write(DATA / "salt_all.jsonl", allrows)

    random.shuffle(train)
    write(DATA / "train.jsonl", train)
    stats["train_total"] = len(train)
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
