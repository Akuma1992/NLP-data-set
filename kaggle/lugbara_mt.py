"""Lugbara machine translation with the models of Akuma (2025), trained on Sunbird SALT.

Thesis: C. D. Akuma, "An Optimized Machine Translation Model for Low-Resource Lugbara
Language", MSc Computer Science, Muni University, Oct 2025.
  * English -> Lugbara : XLM-R (encoder, paired with a warm-started XLM-R decoder)
  * Lugbara -> English : mT5
The thesis corpus (15,524 pairs) is not public, so the same number of sentences is drawn at
random from SALT and split 70/15/15 like the thesis. Luganda uses the same sentences.

Subcommands (run in order; see the Kaggle notebooks):
  prepare  --model M            sample + split SALT, write data/*.jsonl
  grid     --model M            learning-rate x batch-size grid search (thesis 3.4.3)
  train    --model M            fine-tune with the best grid setting, early stopping on val loss
  evaluate --model M            BLEU / TER / chrF / word-F1 / apostrophe integrity on the test split
  analyze                       Lugbara language analysis -> LUGBARA_ANALYSIS.md
  card     --model M            README.md model card
  push     --model M --repo R   upload the trained model, card and analysis
  copy     --repo R             copy mmuniversity/rutooro-multilingual-translator into R
  smoke    --model M --base DIR tiny synthetic run on CPU to check the code end to end
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path

# one GPU only: Kaggle's "T4 x2" would otherwise split each batch over two GPUs and double
# the effective batch size, which would distort the thesis's batch-size grid
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORK = Path(os.environ.get("LUGBARA_WORK", "/kaggle/working" if Path("/kaggle").exists() else "work"))
DATA = WORK / "data"
OUT = WORK / "out"
SALT = "Sunbird/salt"
SOURCE_REPO = "mmuniversity/rutooro-multilingual-translator"
SEED = 13

# thesis Table 3.1: 15,524 pairs, 70/15/15
N_TOTAL, N_TRAIN, N_VAL = 15524, 10867, 2329          # test = 2,328
LOWERCASE = True                                       # thesis 3.2.2: all text lowercased

COLS = {"english": "eng_source_text", "lugbara": "lgg_text", "luganda": "lug_text"}
ISO = {"english": "en", "lugbara": "lgg", "luganda": "lg"}          # Hub metadata codes
# prompt tags; 3-letter codes so Lugbara (lgg) and Luganda (lug) can't be confused
TAG = {"english": "en", "lugbara": "lgg", "luganda": "lug"}

MODELS = {
    "xlmr": {
        "base": "FacebookAI/xlm-roberta-base",
        "title": "XLM-R English→Lugbara/Luganda translator",
        "directions": [("english", "lugbara"), ("english", "luganda")],
        "grid_lr": [3e-5, 5e-5, 1e-4], "grid_bs": [16, 32],
        "lr": 5e-5, "bs": 16, "optim": "adamw_torch", "fp16": True,
        "license": "mit",
        # v2 (v1 reached BLEU 0.3 and looped): longer grid probes, the randomly initialised
        # cross-attention learns 10x faster, label smoothing, more patience, no repeat loops
        "grid_epochs": 2, "patience": 5, "label_smoothing": 0.1,
        "lr_mult": {"crossattention": 10.0},
        "gen": {"no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
        "notes": ["Version 2: grid settings compared after 2 passes instead of 1; the cross-attention "
                  "layers (new, randomly initialised) learn at 10x the grid learning rate; label "
                  "smoothing 0.1; early-stopping patience 5; repeated 3-word phrases are blocked when "
                  "generating. Version 1 (1-pass grid, patience 3, no repeat blocking) stopped at "
                  "validation loss 4.46 and scored BLEU 0.29 / 0.16 because its output looped."],
    },
    "mt5": {
        "base": "google/mt5-base",
        "title": "mT5 Lugbara/Luganda→English translator",
        "directions": [("lugbara", "english"), ("luganda", "english")],
        "grid_lr": [3e-4, 5e-4, 1e-3], "grid_bs": [8, 16],
        "lr": 5e-4, "bs": 16, "optim": "adafactor", "fp16": False,   # mT5 overflows in fp16
        "license": "apache-2.0",
        "grid_epochs": 1, "patience": 3, "label_smoothing": 0.0, "lr_mult": {}, "gen": {}, "notes": [],
    },
}
MAX_LEN = 128

# thesis Table 4.12, on the thesis corpus (shown for reference only)
THESIS = {
    "english-lugbara": {"mBERT": 12.34, "XLM-R": 15.67, "mT5": 18.92, "ULMFiT": 10.15, "XLM-R (tuned)": 22.18},
    "lugbara-english": {"mBERT": 14.57, "XLM-R": 18.92, "mT5": 21.05, "ULMFiT": 11.89, "mT5 (tuned)": 25.41},
}

# ---------------------------------------------------------------- text
APOS = re.compile("[’‘ʼʻ`´′ꞌꞋ]")   # incl. saltillo ꞌ, used in SALT Lugbara
WS = re.compile(r"\s+")
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]*)*|'[^\W\d_]+", re.UNICODE)


def normalize(t):
    if t is None:
        return ""
    t = unicodedata.normalize("NFC", str(t))
    t = APOS.sub("'", t)                       # Lugbara 'b 'd 'y 'w: one apostrophe character
    t = t.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    t = WS.sub(" ", t).strip()
    return t.lower() if LOWERCASE else t


def words(t):
    return WORD.findall(t.lower())


def prefix(tgt):
    """Target-language tag put in front of the source sentence (thesis-style <2xx> tags)."""
    return f"<2{TAG[tgt]}> "


def jl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def jl_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ---------------------------------------------------------------- prepare
def load_salt_rows(fake=0):
    if fake:
        rng = random.Random(SEED)
        syl = ["a", "'ba", "ri", "ma", "ni", "'di", "ku", "ta", "le", "'yo", "si", "ga", "ru", "o", "e"]
        mk = lambda n: " ".join("".join(rng.choice(syl) for _ in range(rng.randint(1, 3))) for _ in range(n))
        return [{"id": i, "split": "train", "eng_source_text": mk(rng.randint(4, 12)),
                 "lgg_text": mk(rng.randint(4, 12)), "lug_text": mk(rng.randint(3, 9)), "raw_lgg": None}
                for i in range(fake)]
    from datasets import load_dataset
    ds = load_dataset(SALT, "text-all")
    rows = []
    for split in ds:
        for r in ds[split]:
            rows.append({"id": r["id"], "split": split, **{c: r[c] for c in COLS.values()},
                         "raw_lgg": r["lgg_text"]})
    return rows


def prepare(args):
    rows = load_salt_rows(args.fake)
    stats = {"salt_rows": len(rows),
             "salt_nonempty": {k: sum(1 for r in rows if normalize(r[c])) for k, c in COLS.items()}}
    clean, seen, dropped = [], set(), Counter()
    for r in rows:
        t = {k: normalize(r[c]) for k, c in COLS.items()}
        if not all(t.values()):
            dropped["empty"] += 1
            continue
        key = (t["english"], t["lugbara"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        lens = [max(1, len(t[k].split())) for k in COLS]
        if max(lens) / min(lens) > 3.5 or max(lens) > 100:
            dropped["malformed"] += 1
            continue
        seen.add(key)
        clean.append({"id": r["id"], "salt_split": r["split"], **t, "raw_lgg": r.get("raw_lgg")})
    rng = random.Random(SEED)
    rng.shuffle(clean)
    n_total = min(N_TOTAL, len(clean)) if not args.fake else len(clean)
    n_train = round(n_total * 0.70) if args.fake else N_TRAIN
    n_val = round(n_total * 0.15) if args.fake else N_VAL
    sample = clean[:n_total]
    splits = {"train": sample[:n_train], "val": sample[n_train:n_train + n_val], "test": sample[n_train + n_val:]}
    for name, part in splits.items():
        jl_write(DATA / f"{name}.jsonl", part)
    stats.update({"clean_rows": len(clean), "dropped": dict(dropped), "seed": SEED,
                  "split_sizes": {k: len(v) for k, v in splits.items()}, "lowercase": LOWERCASE})
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def pairs(split, directions, limit=None):
    rows = jl_read(DATA / f"{split}.jsonl")
    out = []
    for src, tgt in directions:
        sel = rows[:limit] if limit else rows
        out += [{"dir": f"{src}-{tgt}", "src": prefix(tgt) + r[src], "tgt": r[tgt], "id": r["id"]} for r in sel]
    return out


# ---------------------------------------------------------------- model
def build(model_key, base=None, from_dir=None):
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
    cfg = MODELS[model_key]
    base = base or cfg["base"]
    if from_dir:
        tok = AutoTokenizer.from_pretrained(from_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(from_dir)
        return tok, model
    tok = AutoTokenizer.from_pretrained(base)
    if model_key == "xlmr":
        # warm-start an encoder-decoder from XLM-R (Rothe et al., 2020); encoder and decoder
        # share weights, which halves the parameters and fits a 16 GB GPU
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(base, base, tie_encoder_decoder=True)
        model.config.decoder_start_token_id = tok.bos_token_id
        model.config.bos_token_id = tok.bos_token_id
        model.config.eos_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        g = model.generation_config
        g.decoder_start_token_id, g.bos_token_id = tok.bos_token_id, tok.bos_token_id
        g.eos_token_id, g.pad_token_id = tok.eos_token_id, tok.pad_token_id
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(base)
    g = model.generation_config
    g.max_length, g.num_beams, g.early_stopping = MAX_LEN, 4, True
    for k, v in cfg.get("gen", {}).items():
        setattr(g, k, v)
    return tok, model


def group_lr_trainer_class(base_cls):
    """Trainer whose optimiser gives parameter groups their own learning rates.

    `lr_mult` maps a name fragment to a multiplier of the base learning rate (first match wins);
    other trainable parameters use the base rate. Bias/LayerNorm get no weight decay.
    """
    class GroupLRTrainer(base_cls):
        lr_mult = {}
        optimizer_kind = "adamw"

        def create_optimizer(self):
            if self.optimizer is not None or not self.lr_mult:
                return super().create_optimizer()
            import torch
            from transformers.optimization import Adafactor
            groups = {}
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                mult = next((m for frag, m in self.lr_mult.items() if frag in n), 1.0)
                no_decay = n.endswith(".bias") or "LayerNorm" in n or "layer_norm" in n
                groups.setdefault((mult, no_decay), []).append(p)
            lr = self.args.learning_rate
            pg = [{"params": ps, "lr": lr * m, "weight_decay": 0.0 if nd else self.args.weight_decay}
                  for (m, nd), ps in groups.items()]
            if self.optimizer_kind == "adafactor":
                self.optimizer = Adafactor(pg, lr=lr, scale_parameter=False, relative_step=False,
                                           warmup_init=False)
            else:
                self.optimizer = torch.optim.AdamW(pg, lr=lr)
            print("optimizer groups:", [(round(g["lr"], 8), sum(p.numel() for p in g["params"]))
                                        for g in pg], flush=True)
            return self.optimizer
    return GroupLRTrainer


def encode_fn(tok, model_key):
    def enc(b):
        x = tok(b["src"], max_length=MAX_LEN, truncation=True)
        y = tok(text_target=b["tgt"], max_length=MAX_LEN, truncation=True)["input_ids"]
        if model_key == "xlmr":
            y = [ids[1:] for ids in y]          # drop <s>: the decoder starts from it already
        x["labels"] = y
        return x
    return enc


def make_trainer(model_key, tok, model, train_rows, val_rows, lr, bs, out_dir, epochs, cpu=False,
                 early_stop=True, max_steps=-1, save=True):
    from datasets import Dataset
    from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)
    cfg = MODELS[model_key]
    enc = encode_fn(tok, model_key)
    tr = Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in train_rows]).map(enc, batched=True, remove_columns=["src", "tgt"])
    va = Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in val_rows]).map(enc, batched=True, remove_columns=["src", "tgt"])
    args = Seq2SeqTrainingArguments(
        # validation = loss only, small batches: keeping full-vocabulary (250k) logits for the
        # whole validation set ran a 16 GB T4 out of memory
        output_dir=str(out_dir), per_device_train_batch_size=bs, per_device_eval_batch_size=8,
        prediction_loss_only=True,
        learning_rate=lr, optim=cfg["optim"], weight_decay=0.01, warmup_ratio=0.05,
        lr_scheduler_type="linear", num_train_epochs=epochs, max_steps=max_steps,
        fp16=cfg["fp16"] and not cpu, use_cpu=cpu, eval_strategy="epoch",
        save_strategy="epoch" if save else "no",
        save_total_limit=2, load_best_model_at_end=early_stop, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=50, report_to="none", seed=SEED,
        label_smoothing_factor=cfg.get("label_smoothing", 0.0),
        group_by_length=True, dataloader_num_workers=2 if not cpu else 0)
    cbs = [EarlyStoppingCallback(early_stopping_patience=cfg.get("patience", 3))] if early_stop else []
    trainer_cls = group_lr_trainer_class(Seq2SeqTrainer)
    trainer_cls.lr_mult = cfg.get("lr_mult", {})
    trainer_cls.optimizer_kind = "adafactor" if cfg["optim"] == "adafactor" else "adamw"
    return trainer_cls(model=model, args=args, train_dataset=tr, eval_dataset=va,
                       data_collator=DataCollatorForSeq2Seq(tok, model=model), callbacks=cbs,
                       processing_class=tok)


def translate(tok, model, srcs, bs=32, beams=4):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(srcs), bs):
        x = tok(srcs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(dev)
        with torch.no_grad():
            y = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                               num_beams=beams, max_new_tokens=MAX_LEN)
        out += [t.strip() for t in tok.batch_decode(y, skip_special_tokens=True)]
    return out


# ---------------------------------------------------------------- metrics
def word_f1(hyps, refs):
    tp = fp = fn = 0
    for h, r in zip(hyps, refs):
        hc, rc = Counter(words(h)), Counter(words(r))
        common = sum((hc & rc).values())
        tp += common
        fp += sum(hc.values()) - common
        fn += sum(rc.values()) - common
    p = tp / max(1, tp + fp)
    rcl = tp / max(1, tp + fn)
    return round(2 * p * rcl / max(1e-9, p + rcl), 4)


def apostrophe_integrity(hyps, refs):
    """Share of apostrophe words in the reference ('ba, a'dule ...) reproduced intact."""
    hit = tot = 0
    for h, r in zip(hyps, refs):
        hw = set(words(h))
        for w in words(r):
            if "'" in w:
                tot += 1
                hit += w in hw
    return round(hit / tot, 4) if tot else None


def score(hyps, refs, tgt):
    import sacrebleu
    res = {"n": len(refs),
           "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
           "ter": round(sacrebleu.corpus_ter(hyps, [refs]).score, 2),
           "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
           "word_f1": word_f1(hyps, refs)}
    if tgt == "lugbara":
        res["apostrophe_integrity"] = apostrophe_integrity(hyps, refs)
    return res


# ---------------------------------------------------------------- grid / train / evaluate
def device_is_cpu():
    import torch
    return not torch.cuda.is_available()


def grid(args):
    import torch
    cfg = MODELS[args.model]
    train_rows = pairs("train", cfg["directions"])
    val_rows = pairs("val", cfg["directions"])
    lugbara_dir = [d for d in cfg["directions"] if "lugbara" in d][0]
    probe = [r for r in val_rows if r["dir"] == f"{lugbara_dir[0]}-{lugbara_dir[1]}"][:args.probe]
    results = []
    for lr in cfg["grid_lr"]:
        for bs in cfg["grid_bs"]:
            t0 = time.time()
            tok, model = build(args.model, args.base)
            tr = make_trainer(args.model, tok, model, train_rows, val_rows, lr, bs,
                              OUT / args.model / "grid_tmp", epochs=args.grid_epochs or cfg.get("grid_epochs", 1),
                              cpu=device_is_cpu(), early_stop=False, max_steps=args.max_steps, save=False)
            tr.train()
            hyps = translate(tok, model, [r["src"] for r in probe], beams=1)
            bleu = score(hyps, [r["tgt"] for r in probe], lugbara_dir[1])["bleu"]
            ev = tr.evaluate()
            results.append({"lr": lr, "batch_size": bs, "val_bleu": bleu, "val_loss": round(ev["eval_loss"], 4),
                            "minutes": round((time.time() - t0) / 60, 1)})
            print("GRID", results[-1], flush=True)
            del tr, model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    best = max(results, key=lambda r: r["val_bleu"])
    (OUT / args.model).mkdir(parents=True, exist_ok=True)
    (OUT / args.model / "grid.json").write_text(json.dumps({"results": results, "best": best}, indent=2))
    shutil.rmtree(OUT / args.model / "grid_tmp", ignore_errors=True)
    print("BEST", best)


def train(args):
    cfg = MODELS[args.model]
    gpath = OUT / args.model / "grid.json"
    lr, bs = cfg["lr"], cfg["bs"]
    if gpath.exists() and not args.no_grid:
        best = json.loads(gpath.read_text())["best"]
        lr, bs = best["lr"], best["batch_size"]
    tok, model = build(args.model, args.base)
    tr = make_trainer(args.model, tok, model, pairs("train", cfg["directions"]), pairs("val", cfg["directions"]),
                      lr, bs, OUT / args.model / "checkpoints", epochs=args.epochs, cpu=device_is_cpu(),
                      max_steps=args.max_steps)
    t0 = time.time()
    tr.train()
    final = OUT / args.model / "model"
    tr.save_model(str(final))
    tok.save_pretrained(str(final))
    model.generation_config.save_pretrained(str(final))
    (OUT / args.model / "train_log.json").write_text(json.dumps(
        {"lr": lr, "batch_size": bs, "minutes": round((time.time() - t0) / 60, 1),
         "best_checkpoint": tr.state.best_model_checkpoint, "global_step": tr.state.global_step,
         "history": tr.state.log_history}, indent=1))
    shutil.rmtree(OUT / args.model / "checkpoints", ignore_errors=True)
    print("saved", final)


def evaluate(args):
    import torch
    cfg = MODELS[args.model]
    tok, model = build(args.model, from_dir=OUT / args.model / "model")
    if torch.cuda.is_available():
        model = model.cuda()
    res = {}
    for src, tgt in cfg["directions"]:
        rows = [r for r in pairs("test", [(src, tgt)], limit=args.limit)]
        hyps = translate(tok, model, [r["src"] for r in rows])
        d = f"{src}-{tgt}"
        res[d] = score(hyps, [r["tgt"] for r in rows], tgt)
        print(d, res[d], flush=True)
        jl_write(OUT / args.model / f"test_outputs.{d}.jsonl",
                 [{"id": r["id"], "src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in zip(rows, hyps)])
    (OUT / args.model / "scores.json").write_text(json.dumps(res, indent=2))


# ---------------------------------------------------------------- analysis
def corpus_stats(rows, key):
    toks = [w for r in rows for w in words(r[key])]
    lens = [len(words(r[key])) for r in rows]
    c = Counter(toks)
    return {"sentences": len(lens), "words": len(toks), "distinct": len(c),
            "ttr": round(len(c) / max(1, len(toks)), 4),
            "hapax_share": round(sum(1 for v in c.values() if v == 1) / max(1, len(c)), 3),
            "mean_len": round(sum(lens) / max(1, len(lens)), 2),
            "vowel_final": round(sum(v for w, v in c.items() if w[-1] in "aeiou") / max(1, len(toks)), 3),
            "_c": c}


def analyze(args):
    from transformers import AutoTokenizer
    rows = jl_read(DATA / "train.jsonl") + jl_read(DATA / "val.jsonl") + jl_read(DATA / "test.jsonl")
    stats = json.loads((DATA / "stats.json").read_text())
    per = {k: corpus_stats(rows, k) for k in COLS}
    lg = per["lugbara"]
    res = {"stats": stats, "corpus": {k: {kk: vv for kk, vv in v.items() if kk != "_c"} for k, v in per.items()},
           "top_words": lg["_c"].most_common(40)}
    raw = "".join(r.get("raw_lgg") or "" for r in rows)
    res["apostrophe_variants_raw"] = {f"U+{ord(k):04X} {unicodedata.name(k, '?')}": v
                                      for k, v in Counter(ch for ch in raw if ch in "'’‘ʼʻ`´′ꞌꞋ").items()}
    text = " ".join(r["lugbara"] for r in rows)
    res["apostrophe_followed_by"] = Counter(m.group(1) for m in re.finditer(r"'([a-z])", text)).most_common(10)
    res["apostrophe_word_share"] = round(sum(v for w, v in lg["_c"].items() if "'" in w) / max(1, lg["words"]), 3)
    res["non_ascii_letters"] = {ch: n for ch, n in Counter(ch for ch in text if ch.isalpha() and ord(ch) > 127).most_common()}
    onsets = Counter()
    for w, v in lg["_c"].items():
        m = re.match(r"'?([^aeiou']+)", w)
        if m and len(m.group(1)) >= 2:
            onsets[m.group(1)] += v
    res["initial_clusters"] = onsets.most_common(15)
    top = lambda k: {w for w, _ in per[k]["_c"].most_common(2000)}
    res["overlap_top2000"] = {"lugbara~english": round(len(top("lugbara") & top("english")) / 2000, 3),
                              "lugbara~luganda": round(len(top("lugbara") & top("luganda")) / 2000, 3)}
    # how each thesis model's tokenizer cuts Lugbara (thesis 3.2.2: keep the apostrophe inside the word)
    res["tokenizers"] = {}
    apos_words = [w for w, _ in lg["_c"].most_common() if "'" in w][:500]
    for key in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(MODELS[key]["base"])
        except Exception as e:  # offline smoke runs
            print("tokenizer", key, "skipped:", e)
            continue
        fit = {}
        for lang in COLS:
            sample = [r[lang] for r in rows[:3000]]
            pieces = sum(len(tok.tokenize(s)) for s in sample)
            fit[lang] = round(pieces / max(1, sum(len(s.split()) for s in sample)), 2)
        split_apos = 0
        for w in apos_words:
            pcs = [p.lstrip("▁") for p in tok.tokenize(w)]
            split_apos += any(p == "'" for p in pcs)
        res["tokenizers"][key] = {"pieces_per_word": fit,
                                  "apostrophe_cut_off_share": round(split_apos / max(1, len(apos_words)), 3)}
    # words-per-sentence distribution (capped at 40) for the length chart
    res["length_hist"] = {}
    for lang in COLS:
        cnt = Counter(min(40, len(words(r[lang]))) for r in rows)
        res["length_hist"][lang] = {str(n): cnt.get(n, 0) for n in range(1, 41)}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import lugbara_plots
        res["figures"] = [p.name for p in lugbara_plots.data_figures(res, OUT / "figures")]
    except Exception as e:   # plotting must never stop a training run
        print("data figures skipped:", type(e).__name__, e)
        res["figures"] = []
    (OUT / "lugbara_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "LUGBARA_ANALYSIS.md").write_text(render_analysis(res), encoding="utf-8")
    print("wrote", OUT / "LUGBARA_ANALYSIS.md")


def md_table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_analysis(r):
    c, s = r["corpus"], r["stats"]
    figs = set(r.get("figures") or [])
    fig = lambda name, alt: [f"![{alt}](figures/{name})", ""] if name in figs else []
    L = ["# Lugbara language analysis", "",
         "Measured on the 15,524 SALT sentences used to train these models (the thesis-sized sample "
         "described in the model card). Produced by `lugbara_mt.py analyze`.", "",
         "## Background", "",
         "Lugbara (ISO 639-3 `lgg`, Glottolog `lugb1240`) is a Central Sudanic language of the West "
         "Nile region of Uganda and the neighbouring north-east of the DR Congo. It is unrelated to "
         "Luganda and the other Bantu languages of Uganda. Lugbara is tonal, but tone is not written "
         "in everyday spelling. The apostrophe writes a glottal stop and forms the implosive and "
         "pre-glottalised consonants `'b`, `'d`, `'y` and `'w` (Akuma, 2025, §2.3.3).", "",
         "## The sample", "",
         md_table([["SALT rows (all splits)", s["salt_rows"]],
                   ["Rows with Lugbara text", s["salt_nonempty"]["lugbara"]],
                   ["Rows with English, Lugbara and Luganda, after cleaning", s["clean_rows"]],
                   ["Removed (empty / duplicate / malformed)",
                    " / ".join(str(s["dropped"].get(k, 0)) for k in ("empty", "duplicate", "malformed"))],
                   ["Sampled (seed 13)", sum(s["split_sizes"].values())],
                   ["Train / validation / test", " / ".join(str(s["split_sizes"][k]) for k in ("train", "val", "test"))]],
                  ["", "Sentences"]), "",
         "## Sentence length and vocabulary", "",
         md_table([[k.title(), v["sentences"], v["words"], v["distinct"], v["mean_len"], v["ttr"], v["hapax_share"], v["vowel_final"]]
                   for k, v in c.items()],
                  ["Language", "Sentences", "Words", "Distinct words", "Words per sentence", "Type/token",
                   "Share of words seen once", "Word tokens ending in a vowel"]), "",
         f"The same sentences take **{c['lugbara']['mean_len']} words** in Lugbara, "
         f"{c['english']['mean_len']} in English and {c['luganda']['mean_len']} in Luganda.", "",
         *fig("data_sentence_length.png", "Words per sentence in English, Lugbara and Luganda"),
         "### Most frequent Lugbara words", "",
         *fig("data_top_lugbara_words.png", "Most frequent Lugbara words"),
         ", ".join(f"`{w}` ({n})" for w, n in r["top_words"]), "",
         "## Spelling", "",
         *fig("data_apostrophe_letters.png", "Letters following the Lugbara apostrophe"),
         f"* Word tokens containing an apostrophe: **{100 * r['apostrophe_word_share']:.1f}%**",
         "* Apostrophe followed by: " + ", ".join(f"`'{k}` {v}" for k, v in r["apostrophe_followed_by"]),
         "* Apostrophe characters in the raw SALT text (all folded to `'`): "
         + (", ".join(f"{k}: {v}" for k, v in r["apostrophe_variants_raw"].items()) or "none"),
         "* Letters outside ASCII: " + (", ".join(f"`{k}` {v}" for k, v in r["non_ascii_letters"].items()) or "none"),
         "* Most common word-initial consonant clusters: " + ", ".join(f"`{k}` {v}" for k, v in r["initial_clusters"]), "",
         "## Relationship to English and Luganda", "",
         md_table([[k.replace("~", " vs ").title(), f"{100 * v:.1f}%"] for k, v in r["overlap_top2000"].items()],
                  ["Pair", "Shared top-2,000 words"]), "",
         "## How the thesis models' tokenizers cut Lugbara", ""]
    if r["tokenizers"]:
        L += fig("data_tokenizer_pieces.png", "Sub-word pieces per word for each tokenizer")
        L += [md_table([["XLM-R" if k == "xlmr" else "mT5", *[v["pieces_per_word"].get(l, "") for l in COLS],
                         f"{100 * v['apostrophe_cut_off_share']:.1f}%"] for k, v in r["tokenizers"].items()],
                       ["Tokenizer", "Pieces/word English", "Pieces/word Lugbara", "Pieces/word Luganda",
                        "Apostrophe words where `'` becomes its own piece"]), "",
              "The thesis kept the apostrophe inside the word when tokenising (§3.2.2). The last column "
              "shows how often each pre-trained tokenizer instead cuts it off, which is one reason the "
              "model card also reports apostrophe integrity of the output.", ""]
    L += ["## Limits", "",
          "* SALT sentences are short and mostly about agriculture, health and community life; the "
          "thesis corpus also covered tourism and religion.",
          "* Tone is not written, so words that differ only in tone look the same.",
          "* All text is lowercased, as in the thesis.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- card / push / copy
def card(args):
    cfg = MODELS[args.model]
    scores = json.loads((OUT / args.model / "scores.json").read_text())
    tlog = json.loads((OUT / args.model / "train_log.json").read_text())
    gpath = OUT / args.model / "grid.json"
    grid_res = json.loads(gpath.read_text())["results"] if gpath.exists() else []
    stats = json.loads((DATA / "stats.json").read_text())
    repo = args.repo
    langs = sorted({ISO[l] for d in cfg["directions"] for l in d})
    ex_src, ex_tgt = cfg["directions"][0]
    rows = []
    for d, v in scores.items():
        ref = THESIS.get(d, {})
        best_ref = max(ref.items(), key=lambda kv: kv[1]) if ref else None
        rows.append([d.replace("-", " → ").title(), v["n"], v["bleu"], v["ter"], v["word_f1"], v["chrf"],
                     v.get("apostrophe_integrity", "–"),
                     f"{best_ref[1]} ({best_ref[0]})" if best_ref else "–"])
    first_test = jl_read(OUT / args.model / f"test_outputs.{ex_src}-{ex_tgt}.jsonl")[:8]
    usage = "\n".join(f'print(translate("{r["src"].split(" ", 1)[1]}", "{r["src"].split(" ", 1)[0]}"))'
                      for r in first_test[:2])
    notes = "".join(f"* {n}\n" for n in cfg.get("notes", []))
    figdir = OUT / args.model / "figures"
    figs_md = "".join(f"![{alt}](figures/{name})\n\n" for name, alt in (
        ("test_scores.png", "Test scores"), ("quality_by_length.png", "Quality by sentence length"),
        ("training.png", "Training and validation loss"), ("grid_search.png", "Grid search"))
        if (figdir / name).exists())
    md = f"""---
language: [{", ".join(langs)}]
license: {cfg['license']}
library_name: transformers
pipeline_tag: translation
base_model: {cfg['base']}
datasets: [Sunbird/salt]
metrics: [bleu, ter, chrf]
tags: [translation, lugbara, luganda, low-resource, uganda, muni-university]
---

# {cfg['title']}

Fine-tuned `{cfg['base']}` for {", ".join(f"{a.title()} → {b.title()}" for a, b in cfg['directions'])}.
It follows the model choice and training method of **C. D. Akuma, *An Optimized Machine
Translation Model for Low-Resource Lugbara Language*, MSc Computer Science, Muni University
(October 2025)**: {"XLM-R was the thesis's optimised English→Lugbara model" if args.model == "xlmr" else "mT5 was the thesis's optimised Lugbara→English model"}.

## How to use

Put the target-language tag in front of the sentence and use lowercase text, as in training.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSeq2SeqLM.from_pretrained("{repo}")

def translate(text, tag):
    x = tok(tag + " " + text.lower(), return_tensors="pt")
    y = model.generate(**x, num_beams=4, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)

{usage}
```

Tags: {", ".join(f"`{prefix(t).strip()}` for {s.title()} → {t.title()}" for s, t in cfg["directions"])}.

## Results (test split, {sum(v["n"] for v in scores.values())} sentence pairs)

{md_table(rows, ["Direction", "Sentences", "BLEU ↑", "TER ↓", "Word F1 ↑", "chrF ↑", "Apostrophe integrity ↑", "Thesis best BLEU*"])}

BLEU and TER are sacreBLEU corpus scores; word F1 is unigram precision/recall F1; apostrophe
integrity is the share of apostrophe words in the reference (`'ba`, `a'dule` …) reproduced intact.
\\*The thesis scores were measured on the thesis's own corpus, not on SALT, so they are a
reference point rather than a like-for-like comparison.

{figs_md}
### Sample test translations ({ex_src.title()} → {ex_tgt.title()})

{md_table([[r["src"].split(" ", 1)[1], r["ref"], r["hyp"]] for r in first_test], ["Source", "Reference", "Model"])}

## Training

* **Data:** the thesis corpus (15,524 English–Lugbara pairs) is not public, so the same number
  of sentences was drawn at random (seed {SEED}) from the cleaned [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt)
  `text-all` table and split 70/15/15 like the thesis: {stats['split_sizes']['train']:,} train,
  {stats['split_sizes']['val']:,} validation, {stats['split_sizes']['test']:,} test. Luganda uses
  the same sentences. SALT has no domain labels, so the split is random rather than stratified by domain.
* **Preprocessing (thesis §3.2.2):** lowercased, punctuation normalised, all apostrophe variants
  folded to `'`, empty / duplicate / malformed pairs removed.
* **Model:** {"encoder-decoder warm-started from XLM-R base (Rothe et al., 2020), encoder and decoder weights shared" if args.model == "xlmr" else "mT5 base, sequence-to-sequence"}.
* **Hyperparameters (thesis §3.4.3):** grid search over learning rate × batch size
  ({", ".join(f"{g['lr']:g}/{g['batch_size']} → {g['val_bleu']} BLEU" for g in grid_res) or "not run"}),
  then fine-tuning with lr {tlog['lr']:g}, batch {tlog['batch_size']}, optimiser {cfg['optim']},
  early stopping on validation loss (patience {cfg.get('patience', 3)}); {tlog['global_step']:,} steps,
  {tlog['minutes']} minutes on a Kaggle GPU.

{notes}
See [LUGBARA_ANALYSIS.md](LUGBARA_ANALYSIS.md) for the Lugbara language analysis of the training data.

## Limits

* Trained on short SALT sentences (agriculture, health, community life); other domains are weaker.
* Lowercase only. Tone is not written in Lugbara, so tone-only differences are lost.
* Automatic scores use one reference per sentence; Lugbara speakers should review output before use.

## Credits

* Thesis and method: Charity Driciru Akuma, Muni University (2025).
* Data: Sunbird AI, SALT (CC BY-SA 4.0).
* Base model: `{cfg['base']}`.
* Earlier work: [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}) (Mountains of the Moon University).
"""
    (OUT / args.model / "model" / "README.md").write_text(md, encoding="utf-8")
    print("wrote card for", repo)


def plots(args):
    """Grid, training, test-score and length figures for one model (shown inline by the notebook)."""
    import lugbara_plots
    paths = lugbara_plots.model_figures(OUT / args.model, MODELS[args.model]["title"].split(" ")[0], THESIS)
    for p in paths:
        print("figure", p)


def copy_figures_and_outputs(model_out, folder):
    """Data figures + this model's figures -> figures/, test outputs -> test_outputs/."""
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    for src in (OUT / "figures", model_out / "figures"):
        for p in src.glob("*.png") if src.exists() else []:
            shutil.copy(p, folder / "figures" / p.name)
    (folder / "test_outputs").mkdir(exist_ok=True)
    for p in model_out.glob("test_outputs.*.jsonl"):
        shutil.copy(p, folder / "test_outputs" / p.name)


def push(args):
    from huggingface_hub import HfApi
    api = HfApi()
    folder = OUT / args.model / "model"
    for f in ("LUGBARA_ANALYSIS.md", "lugbara_analysis.json"):
        if (OUT / f).exists():
            shutil.copy(OUT / f, folder / f)
    for f in ("scores.json", "train_log.json", "grid.json"):
        if (OUT / args.model / f).exists():
            shutil.copy(OUT / args.model / f, folder / f)
    copy_figures_and_outputs(OUT / args.model, folder)
    api.create_repo(args.repo, exist_ok=True, private=False)
    # replace whatever was there before (e.g. the Rutooro copy), keep .gitattributes
    api.upload_folder(folder_path=str(folder), repo_id=args.repo,
                      commit_message=f"{MODELS[args.model]['title']} trained on SALT (thesis method)",
                      delete_patterns=["*.safetensors", "*.bin", "*.spm", "*.model", "*.json", "*.md", "*.txt"])
    print("pushed", f"https://huggingface.co/{args.repo}")


def copy(args):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if api.repo_exists(args.repo) and any(f.endswith(".safetensors") for f in api.list_repo_files(args.repo)):
        print("skip copy:", args.repo, "already has a model")
        return
    local = Path(snapshot_download(SOURCE_REPO, local_dir=str(WORK / "rutooro-copy")))
    readme = (local / "README.md").read_text(encoding="utf-8")
    head, body = readme.split("---", 2)[1], readme.split("---", 2)[2]
    note = (f"\n> **Copy of [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO})** (MIT licence, "
            "Mountains of the Moon University), published here as the starting point for Muni "
            "University's Lugbara translators. Lugbara models are being trained and will replace "
            "this copy.\n")
    (local / "README.md").write_text("---" + head + "---\n" + note + body, encoding="utf-8")
    shutil.rmtree(local / ".cache", ignore_errors=True)
    api.create_repo(args.repo, exist_ok=True, private=False)
    api.upload_folder(folder_path=str(local), repo_id=args.repo, commit_message=f"Copy of {SOURCE_REPO}")
    print("copied to", f"https://huggingface.co/{args.repo}")


# ---------------------------------------------------------------- smoke test
def smoke(args):
    """Tiny end-to-end run on CPU with synthetic data: prepare, grid(1 point), train, evaluate, card.
    --base-root holds local copies of the base models (xlm-roberta-base/, mt5-small/)."""
    if args.base_root:
        MODELS["xlmr"]["base"] = str(args.base_root / "xlm-roberta-base")
        MODELS["mt5"]["base"] = str(args.base_root / "mt5-small")
    cfg = MODELS[args.model]
    cfg["grid_lr"], cfg["grid_bs"], cfg["optim"] = cfg["grid_lr"][:1], [2], "adafactor"
    ns = argparse.Namespace(fake=40, model=args.model, base=None, probe=4, grid_epochs=1, max_steps=3,
                            epochs=1, no_grid=False, limit=4, repo="local/smoke")
    prepare(ns)
    grid(ns)
    train(ns)
    evaluate(ns)
    analyze(ns)
    plots(ns)
    card(ns)
    # reload check
    tok, model = build(args.model, from_dir=OUT / args.model / "model")
    print("reload OK:", translate(tok, model, [prefix(cfg["directions"][0][1]) + "a 'ba ri"], beams=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "grid", "train", "evaluate", "analyze", "plots", "card", "push", "copy", "smoke"])
    ap.add_argument("--model", choices=list(MODELS), default="xlmr")
    ap.add_argument("--base", help="local path or hub id overriding the base model")
    ap.add_argument("--base-root", type=Path, help="smoke: folder holding xlmr/ and mt5/ tokenizers")
    ap.add_argument("--repo")
    ap.add_argument("--fake", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--grid-epochs", type=float, default=None, help="default: MODELS[model]['grid_epochs']")
    ap.add_argument("--probe", type=int, default=300, help="validation sentences scored per grid point")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()
    random.seed(SEED)
    {"prepare": prepare, "grid": grid, "train": train, "evaluate": evaluate, "analyze": analyze, "plots": plots,
     "card": card, "push": push, "copy": copy, "smoke": smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
