"""Build train/dev/test files from Sunbird SALT (text-all).

Train = every English->Lugbara pair in SALT train, plus a replay sample of the four
languages the Rutooro model already knows, so fine-tuning on Lugbara doesn't erase them.
"""
import json
import random

from datasets import load_dataset

from common import DATA, LANGS, NEW_LANG, SALT_REPO, normalize, tag

REPLAY_PER_LANG = 2000
SEED = 13


def rows(split, col):
    out = []
    for r in split:
        src, tgt = normalize(r["eng_source_text"]), normalize(r[col])
        if src and tgt:
            out.append({"id": r["id"], "src": src, "tgt": tgt})
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

    train = []
    stats = {}
    for lang, col in LANGS.items():
        tr = rows(ds["train"], col)
        if lang != NEW_LANG:
            random.shuffle(tr)
            tr = tr[:REPLAY_PER_LANG]
        for it in tr:
            train.append({"lang": lang, "src": f"{tag(lang)} {it['src']}", "tgt": it["tgt"]})
        dev = rows(ds["dev"], col)
        test = rows(ds["test"], col)
        write(DATA / f"dev.{lang}.jsonl", dev)
        write(DATA / f"test.{lang}.jsonl", test)
        stats[lang] = {"train": len(tr), "dev": len(dev), "test": len(test)}

    # raw Lugbara side of the whole corpus, for the language analysis
    lgg_all = []
    for split in ("train", "dev", "test"):
        for r in ds[split]:
            t = normalize(r["lgg_text"])
            if t:
                lgg_all.append({"split": split, "id": r["id"], "eng": normalize(r["eng_source_text"]),
                                "lgg": t, "raw": r["lgg_text"],
                                **{k: normalize(r[c]) for k, c in LANGS.items() if k != NEW_LANG}})
    write(DATA / "salt_lugbara_all.jsonl", lgg_all)

    # how much of each SALT table actually has text for each language
    coverage = {}
    for split in ("train", "dev", "test"):
        n = len(ds[split])
        coverage[split] = {"rows": n, **{lang: sum(1 for r in ds[split] if normalize(r[col]))
                                         for lang, col in LANGS.items()}}
    stats["coverage"] = coverage

    random.shuffle(train)
    write(DATA / "train.jsonl", train)
    stats["train_total"] = len(train)
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
