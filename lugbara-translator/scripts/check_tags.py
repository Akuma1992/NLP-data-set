"""Show how a model's tokenizer reads each language tag and what it outputs for each.

Usage:  python scripts/check_tags.py [model-dir]   (default: source-model)
"""
import json
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import ANALYSIS, LANGS, ROOT, tag

SENTENCES = ["Education is important for community development.",
             "The market opens early in the morning."]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "source-model"
    tok = AutoTokenizer.from_pretrained(ROOT / name)
    model = AutoModelForSeq2SeqLM.from_pretrained(ROOT / name).eval()
    out = {"model": name, "tags": {}, "outputs": {}}
    for lang in LANGS:
        ids = tok(f"{tag(lang)} {SENTENCES[0]}")["input_ids"]
        out["tags"][lang] = {"first_id": ids[0], "first_token": tok.convert_ids_to_tokens(ids[0])}
    for s in SENTENCES:
        batch = tok([f"{tag(l)} {s}" for l in LANGS], return_tensors="pt", padding=True)
        with torch.no_grad():
            y = model.generate(**batch, num_beams=4, max_new_tokens=64)
        out["outputs"][s] = dict(zip(LANGS, tok.batch_decode(y, skip_special_tokens=True)))
    out["distinct_outputs_per_sentence"] = {s: len(set(v.values())) for s, v in out["outputs"].items()}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / f"tags.{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
