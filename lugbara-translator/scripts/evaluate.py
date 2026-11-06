"""Score a model on the SALT test split for each language (BLEU + chrF) and save outputs.

Usage:  python scripts/evaluate.py --model source-model --name source [--n 200]
"""
import argparse
import json
import os
import time

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import ANALYSIS, DATA, LANGS, ROOT, tag


def load(path, n):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]
    return items[:n] if n else items


def translate(model, tok, texts, bs=16, beams=4):
    out = []
    for i in range(0, len(texts), bs):
        x = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            y = model.generate(**x, num_beams=beams, max_new_tokens=128)
        out += tok.batch_decode(y, skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=200, help="sentences per language (Lugbara always uses all)")
    ap.add_argument("--langs", default=",".join(LANGS))
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))

    tok = AutoTokenizer.from_pretrained(ROOT / args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(ROOT / args.model).eval()
    ANALYSIS.mkdir(exist_ok=True)
    res = {}
    for lang in args.langs.split(","):
        items = load(DATA / f"test.{lang}.jsonl", 0 if lang == "lugbara" else args.n)
        t0 = time.time()
        hyp = translate(model, tok, [f"{tag(lang)} {it['src']}" for it in items])
        refs = [it["tgt"] for it in items]
        res[lang] = {
            "n": len(items),
            "bleu": round(sacrebleu.corpus_bleu(hyp, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyp, [refs]).score, 2),
            "seconds": round(time.time() - t0),
        }
        print(lang, res[lang], flush=True)
        with open(ANALYSIS / f"outputs.{args.name}.{lang}.jsonl", "w", encoding="utf-8") as f:
            for it, h in zip(items, hyp):
                f.write(json.dumps({"id": it["id"], "src": it["src"], "ref": it["tgt"], "hyp": h},
                                   ensure_ascii=False) + "\n")
    (ANALYSIS / f"scores.{args.name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
