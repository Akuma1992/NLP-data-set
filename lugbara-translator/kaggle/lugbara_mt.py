"""Lugbara machine translation with the models of Akuma (2025), trained on Sunbird SALT.

Thesis: C. D. Akuma, "An Optimized Machine Translation Model for Low-Resource Lugbara
Language", MSc Computer Science, Muni University, Oct 2025.
  * English -> Lugbara : XLM-R (encoder, paired with a warm-started XLM-R decoder)
  * Lugbara -> English : mT5
The thesis corpus (15,524 pairs) is not public, so the same number of sentences is drawn at
random from SALT and split 70/15/15 like the thesis. Luganda uses the same sentences.
The *_local models train the same two models, with the settings their SALT runs chose, on the
local English-Lugbara corpus published with the thesis (github.com/Akuma1992/NLP-data-set),
so the two corpora can be compared.

Subcommands (run in order; see the Kaggle notebooks):
  prepare  [--dataset local]     sample + split SALT -> data/*.jsonl (local corpus -> data_local/)
  grid     --model M            learning-rate x batch-size grid search (thesis 3.4.3)
  train    --model M            fine-tune with the best grid setting, early stopping on val loss
  evaluate --model M            BLEU / TER / chrF / word-F1 / apostrophe integrity on the test split
           [--on salt|local] [--counterpart | --from-repo R --revision V] [--name N]   cross-tests
  compare  --model M            *_local models: trained on SALT vs local corpus x tested on both (2 x 2)
  analyze  [--dataset local]    Lugbara language analysis -> LUGBARA_ANALYSIS.md
  card     --model M            README.md model card
  fetch    --model M --repo R   download a published run's results (no weights) to redo compare + card
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
DATA = WORK / "data"                 # SALT split
DATA_LOCAL = WORK / "data_local"     # local corpus split (Akuma 2025)
OUT = WORK / "out"
LOCAL_URL = "https://raw.githubusercontent.com/Akuma1992/NLP-data-set/main/datasetnlp%20.csv"
LOCAL_SOURCE = "https://github.com/Akuma1992/NLP-data-set"


def data_dir(dataset):
    return DATA_LOCAL if dataset == "local" else DATA
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
    # the same two models trained on the local English-Lugbara corpus of Akuma (2025), with the
    # settings their SALT runs chose (no new grid), so that only the data differs
    "mt5_local": {
        "base": "google/mt5-base", "dataset": "local", "kind": "t5",
        "title": "mT5 Lugbara→English translator (local corpus)",
        "directions": [("lugbara", "english")],
        "lr": 1e-3, "bs": 16, "optim": "adafactor", "fp16": False, "license": "apache-2.0",
        "grid_epochs": 1, "patience": 3, "label_smoothing": 0.0, "lr_mult": {}, "gen": {},
        "counterpart": {"key": "mt5", "repo": "ckuma/lugbara-english-translator", "revision": None},
        "notes": ["Same model and settings as the SALT-trained mT5 (learning rate 1e-3, batch 16, the "
                  "winner of its SALT grid search); only the training data differs."],
    },
    "xlmr_local": {
        "base": "FacebookAI/xlm-roberta-base", "dataset": "local", "kind": "encdec",
        "title": "XLM-R English→Lugbara translator (local corpus)",
        "directions": [("english", "lugbara")],
        "lr": 3e-5, "bs": 16, "optim": "adamw_torch", "fp16": True, "license": "mit",
        "grid_epochs": 1, "patience": 3, "label_smoothing": 0.0, "lr_mult": {}, "gen": {},
        "counterpart": {"key": "xlmr", "repo": "ckuma/lugbara-multilingual-translator", "revision": "b2eec05b"},
        "notes": ["Same model and settings as the first SALT-trained XLM-R (version 1: learning rate 3e-5, "
                  "batch 16, the winner of its SALT grid search); only the training data differs."],
    },
}
for _k, _c in MODELS.items():
    _c.setdefault("dataset", "salt")
    _c.setdefault("kind", "encdec" if _k == "xlmr" else "t5")
MAX_LEN = 128

# thesis Table 4.12, on the thesis corpus (shown for reference only)
THESIS = {
    "english-lugbara": {"mBERT": 12.34, "XLM-R": 15.67, "mT5": 18.92, "ULMFiT": 10.15, "XLM-R (tuned)": 22.18},
    "lugbara-english": {"mBERT": 14.57, "XLM-R": 18.92, "mT5": 21.05, "ULMFiT": 11.89, "mT5 (tuned)": 25.41},
}

# ---------------------------------------------------------------- text
APOS_CHARS = "'’‘ʼʻ`´′ꞌꞋ"
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


def load_local_rows():
    """The local corpus CSV (columns English, Lugbara). Mostly UTF-8 with a few Windows-1252 lines.
    Downloaded from GitHub, or read from the file named by LUGBARA_LOCAL_CSV."""
    import csv
    import io
    import urllib.request
    path = os.environ.get("LUGBARA_LOCAL_CSV")
    if path:
        raw = Path(path).read_bytes()
    else:
        for attempt in range(4):
            try:
                raw = urllib.request.urlopen(LOCAL_URL, timeout=120).read()
                break
            except OSError as e:
                if attempt == 3:
                    raise
                print("download failed, retrying:", e, flush=True)
                time.sleep(10 * (attempt + 1))
    lines = []
    for line in raw.split(b"\n"):
        try:
            lines.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(line.decode("cp1252", errors="replace"))
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = [h.strip().lower() for h in next(reader)]
    ie, il = header.index("english"), header.index("lugbara")
    return [{"id": i, "split": "local", "english": r[ie] if len(r) > ie else "",
             "lugbara": r[il] if len(r) > il else "", "raw_lgg": r[il] if len(r) > il else None}
            for i, r in enumerate(reader)]


def prepare_local(args):
    """Same cleaning rules as SALT; every clean pair is used, split 70/15/15 at random (seed 13)."""
    rows = load_local_rows()
    langs = ("english", "lugbara")
    stats = {"source": LOCAL_SOURCE, "rows": len(rows),
             "nonempty": {k: sum(1 for r in rows if normalize(r[k])) for k in langs}}
    clean, seen, dropped = [], set(), Counter()
    for r in rows:
        t = {k: normalize(r[k]) for k in langs}
        if not all(t.values()):
            dropped["empty"] += 1
            continue
        if (t["english"], t["lugbara"]) in seen:
            dropped["duplicate"] += 1
            continue
        lens = [max(1, len(t[k].split())) for k in langs]
        if max(lens) / min(lens) > 3.5 or max(lens) > 100:
            dropped["malformed"] += 1
            continue
        seen.add((t["english"], t["lugbara"]))
        clean.append({"id": r["id"], **t, "raw_lgg": r["raw_lgg"]})
    random.Random(SEED).shuffle(clean)
    n_train, n_val = round(len(clean) * 0.70), round(len(clean) * 0.15)
    splits = {"train": clean[:n_train], "val": clean[n_train:n_train + n_val], "test": clean[n_train + n_val:]}
    for name, part in splits.items():
        jl_write(DATA_LOCAL / f"{name}.jsonl", part)
    stats.update({"clean_rows": len(clean), "dropped": dict(dropped), "seed": SEED,
                  "split_sizes": {k: len(v) for k, v in splits.items()}, "lowercase": LOWERCASE})
    (DATA_LOCAL / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def prepare(args):
    if getattr(args, "dataset", "salt") == "local":
        return prepare_local(args)
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


def pairs(split, directions, limit=None, dataset="salt"):
    rows = jl_read(data_dir(dataset) / f"{split}.jsonl")
    out = []
    for src, tgt in directions:
        if rows and (src not in rows[0] or tgt not in rows[0]):
            continue          # e.g. Luganda directions on the English-Lugbara local corpus
        sel = rows[:limit] if limit else rows
        out += [{"dir": f"{src}-{tgt}", "src": prefix(tgt) + r[src], "tgt": r[tgt], "id": r["id"]} for r in sel]
    return out


# ---------------------------------------------------------------- model
def build(model_key, base=None, from_dir=None, revision=None):
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
    cfg = MODELS[model_key]
    base = base or cfg["base"]
    if from_dir:
        tok = AutoTokenizer.from_pretrained(str(from_dir), revision=revision)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(from_dir), revision=revision)
        return tok, model
    tok = AutoTokenizer.from_pretrained(base)
    if cfg["kind"] == "encdec":
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
    kind = MODELS[model_key]["kind"]

    def enc(b):
        x = tok(b["src"], max_length=MAX_LEN, truncation=True)
        y = tok(text_target=b["tgt"], max_length=MAX_LEN, truncation=True)["input_ids"]
        if kind == "encdec":
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
    if "grid_lr" not in cfg:
        print(args.model, "has no grid: it reuses the settings its SALT counterpart chose")
        return
    train_rows = pairs("train", cfg["directions"], dataset=cfg["dataset"])
    val_rows = pairs("val", cfg["directions"], dataset=cfg["dataset"])
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
    tr = make_trainer(args.model, tok, model, pairs("train", cfg["directions"], dataset=cfg["dataset"]), pairs("val", cfg["directions"], dataset=cfg["dataset"]),
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
    """Score on a test split. Defaults: this run's model on its own dataset's test split.
    --on salt|local       another test split (cross-test)
    --counterpart         the published SALT-trained counterpart instead of this run's model
    --from-repo/--revision any other published model
    --name N              write to out/MODEL/cross/N/ instead of out/MODEL/"""
    import torch
    cfg = MODELS[args.model]
    on = getattr(args, "on", None) or cfg["dataset"]
    repo, rev = getattr(args, "from_repo", None), getattr(args, "revision", None)
    if getattr(args, "counterpart", False):
        repo, rev = cfg["counterpart"]["repo"], cfg["counterpart"]["revision"]
    tok, model = build(args.model, from_dir=repo or OUT / args.model / "model", revision=rev)
    if torch.cuda.is_available():
        model = model.cuda()
    name = getattr(args, "name", None)
    dest = OUT / args.model / "cross" / name if name else OUT / args.model
    dest.mkdir(parents=True, exist_ok=True)
    res = {}
    for src, tgt in cfg["directions"]:
        rows = pairs("test", [(src, tgt)], limit=args.limit, dataset=on)
        if not rows:
            continue
        hyps = translate(tok, model, [r["src"] for r in rows])
        d = f"{src}-{tgt}"
        res[d] = score(hyps, [r["tgt"] for r in rows], tgt)
        print(f"[{repo or 'this model'}{'@' + rev if rev else ''} on {on} test]", d, res[d], flush=True)
        jl_write(dest / f"test_outputs.{d}.jsonl",
                 [{"id": r["id"], "src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in zip(rows, hyps)])
    (dest / "scores.json").write_text(json.dumps(res, indent=2))
    if name:
        (dest / "about.json").write_text(json.dumps(
            {"model": repo or f"this run ({args.model})", "revision": rev, "test_set": on}, indent=2))


CROSS = {   # (trained on, tested on) -> where the scores are, for a local-corpus run
    ("local", "local"): "scores.json",
    ("local", "salt"): "cross/on_salt/scores.json",
    ("salt", "local"): "cross/salt_model_on_local/scores.json",
    ("salt", "salt"): "cross/salt_model_on_salt/scores.json",
}


def compare(args):
    """Trained on SALT vs trained on the local corpus, each tested on both test splits (2 x 2).
    The SALT model's SALT score is its published one (same split, same code)."""
    cfg = MODELS[args.model]
    return cross_table(OUT / args.model, cfg["counterpart"], cfg["directions"], cfg["title"].split(" ")[0])


def hub_link(cp):
    return f"[{cp['repo']}](https://huggingface.co/{cp['repo']}" + \
           (f"/tree/{cp['revision']})" if cp.get("revision") else ")")


def english_of(row, src, tgt):
    """The English side of a test-output row (reference when translating into English, else the source)."""
    return row["ref"] if tgt == "english" else row["src"].split(" ", 1)[1]


def cross_outputs(d, cell, dname):
    """Test-output file of one 2 x 2 cell (see CROSS) for one direction."""
    return d / Path(CROSS[cell]).parent / f"test_outputs.{dname}.jsonl"


def unshared_table(d, directions):
    """The 2 x 2 again, each test split cut to the sentences whose English is NOT in the other corpus's
    training split, so no cross-test cell can be helped by sentences its model saw in training.
    Needs both splits prepared and every cell's test outputs; returns ({}, {}) otherwise."""
    if not ((DATA / "train.jsonl").exists() and (DATA_LOCAL / "train.jsonl").exists()):
        return {}, {}
    eng_train = {n: {r["english"] for r in jl_read(data_dir(n) / "train.jsonl")} for n in ("salt", "local")}
    other = {"salt": "local", "local": "salt"}
    table, sizes = {}, {}
    for src, tgt in directions:
        dname = f"{src}-{tgt}"
        for cell in CROSS:
            p = cross_outputs(d, cell, dname)
            if not p.exists():
                continue
            test_on = cell[1]
            rows = [r for r in jl_read(p) if english_of(r, src, tgt) not in eng_train[other[test_on]]]
            if rows:
                res = score([r["hyp"] for r in rows], [r["ref"] for r in rows], tgt)
                table.setdefault(dname, {})[f"{cell[0]}>{test_on}"] = res
                sizes.setdefault(dname, {})[test_on] = len(rows)
    return table, sizes


def cross_table(d, cp, directions, model_name):
    """The 2 x 2 table for a local-corpus run whose cross-test scores are under d (see CROSS), on the
    full test splits and on the sentences the two corpora do not share."""
    from huggingface_hub import hf_hub_download
    ss = d / CROSS[("salt", "salt")]
    ss.parent.mkdir(parents=True, exist_ok=True)
    if not ss.exists():
        shutil.copy(hf_hub_download(cp["repo"], "scores.json", revision=cp["revision"]), ss)
        (ss.parent / "about.json").write_text(json.dumps(
            {"model": cp["repo"], "revision": cp["revision"], "test_set": "salt",
             "note": "published test scores of the SALT-trained model"}, indent=2))
    dirs = [f"{a}-{b}" for a, b in directions]
    for dname in dirs:      # the SALT model's own SALT test outputs, when it published them
        f = cross_outputs(d, ("salt", "salt"), dname)
        if not f.exists():
            try:
                shutil.copy(hf_hub_download(cp["repo"], f"test_outputs/test_outputs.{dname}.jsonl",
                                            revision=cp["revision"]), f)
            except Exception as e:
                print(f"no published SALT test outputs for {dname} ({type(e).__name__}); run "
                      "evaluate --counterpart --on salt --name salt_model_on_salt for the unshared table")
    table = {}
    for (train_on, test_on), rel in CROSS.items():
        if (d / rel).exists():
            for dname, v in json.loads((d / rel).read_text()).items():
                if dname in dirs:
                    table.setdefault(dname, {})[f"{train_on}>{test_on}"] = v
    t2, sizes = unshared_table(d, directions)
    out = {"model": d.name, "counterpart": cp, "table": table, "rows": "trained on", "columns": "tested on",
           "table_unshared": t2, "unshared_sentences": sizes,
           "unshared_note": "each test split cut to the sentences whose English is not in the other corpus's training split"}
    (d / "comparison.json").write_text(json.dumps(out, indent=2))
    for name, tb in (("all test sentences", table), ("sentences not shared between the corpora", t2)):
        print(f"--- {name}")
        for dname, cells in tb.items():
            print(dname, sizes.get(dname, "") if tb is t2 else "")
            for k, v in cells.items():
                print(f"  trained on {k.split('>')[0]:5s}  tested on {k.split('>')[1]:5s}  "
                      f"BLEU {v['bleu']:6.2f}  chrF {v['chrf']:6.2f}  (n={v['n']})")
    try:
        import lugbara_plots
        print("figure", lugbara_plots.dataset_figure(out, d / "figures" / "salt_vs_local.png", model_name))
        if t2:
            print("figure", lugbara_plots.dataset_figure(
                {**out, "table": t2}, d / "figures" / "salt_vs_local_unshared.png", model_name,
                subtitle="test sentences the two corpora do not share"))
    except Exception as e:  # plotting must never stop a run
        print("comparison figure skipped:", type(e).__name__, e)
    return out


def cross_section_md(d, cp, what="model"):
    """Model-card section for the 2 x 2 table in d/comparison.json ('' if there is none)."""
    cpath = d / "comparison.json"
    if not cpath.exists():
        return ""
    comp = json.loads(cpath.read_text())
    cell = lambda cells, k: f"{cells[k]['bleu']} / {cells[k]['chrf']}" if k in cells else "–"

    def rows_of(table):
        trows = []
        for dname, cells in table.items():
            trows.append([dname.replace("-", " → ").title(), f"SALT ({hub_link(cp)})",
                          cell(cells, "salt>salt"), cell(cells, "salt>local")])
            trows.append(["", f"Local corpus (this {what})", cell(cells, "local>salt"), cell(cells, "local>local")])
        return trows
    notes = ""
    apath = OUT / "lugbara_analysis.json"
    vs = json.loads(apath.read_text(encoding="utf-8")).get("vs_salt") if apath.exists() else None
    if vs:
        u, same, n = vs["unseen_test_words"]["lugbara"], vs["same_english_sentence"], vs["test_sentences"]
        notes = (f"\n**Shared sentences.** {same['local_in_salt']:,} of the {vs['sentences']['local']:,} local pairs have an "
                 f"English sentence that is also in the SALT sample: {same['local_test_in_salt_train']:,} of the "
                 f"{n['local']:,} local test sentences are in SALT's training split, and "
                 f"{same['salt_test_in_local_train']:,} of the {n['salt']:,} SALT test sentences are in the local "
                 "training split, so the cross-test cells above are partly helped by sentences seen in training. "
                 "The table below repeats the comparison without them.\n\n"
                 f"Lugbara word tokens of each test split that never occur in the training split: "
                 f"SALT test {100 * u['salt>salt']:.1f}% (SALT train) vs {100 * u['local>salt']:.1f}% (local train); "
                 f"local test {100 * u['local>local']:.1f}% (local train) vs {100 * u['salt>local']:.1f}% (SALT train). "
                 "See [LUGBARA_ANALYSIS.md](LUGBARA_ANALYSIS.md).\n")
    fig = "![SALT vs local corpus](figures/salt_vs_local.png)\n" if (d / "figures" / "salt_vs_local.png").exists() else ""
    unshared = ""
    if comp.get("table_unshared"):
        sizes = comp.get("unshared_sentences", {})
        first = next(iter(sizes.values()), {})
        fig2 = ("![SALT vs local corpus, unshared sentences](figures/salt_vs_local_unshared.png)\n"
                if (d / "figures" / "salt_vs_local_unshared.png").exists() else "")
        unshared = f"""
### Only the test sentences the two corpora do not share

Each test split cut to the sentences whose English is not in the other corpus's training split
({first.get('salt', '?'):,} SALT test and {first.get('local', '?'):,} local test sentences), so every cell is scored on
sentences none of the models saw in training.

{md_table(rows_of(comp["table_unshared"]), ["Direction", "Trained on", "SALT test (unshared)", "Local test (unshared)"])}

{fig2}"""
    return f"""
## Trained on SALT vs trained on the local corpus

The same {what} with the same settings, trained once on each corpus and tested on both test splits
(BLEU / chrF; the SALT test split is the {N_TOTAL - N_TRAIN - N_VAL:,}-sentence split of the thesis-sized SALT sample).

{md_table(rows_of(comp["table"]), ["Direction", "Trained on", "SALT test", "Local test"])}

{fig}{notes}{unshared}"""


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
    """Lugbara language analysis of a training corpus (--dataset salt|local) -> LUGBARA_ANALYSIS.md.
    For the local corpus it also compares it with the SALT sample when that is prepared."""
    from transformers import AutoTokenizer
    dataset = getattr(args, "dataset", None) or "salt"
    ddir = data_dir(dataset)
    langs = list(COLS) if dataset == "salt" else ["english", "lugbara"]
    rows = jl_read(ddir / "train.jsonl") + jl_read(ddir / "val.jsonl") + jl_read(ddir / "test.jsonl")
    stats = json.loads((ddir / "stats.json").read_text())
    per = {k: corpus_stats(rows, k) for k in langs}
    lg = per["lugbara"]
    res = {"dataset": dataset, "stats": stats,
           "corpus": {k: {kk: vv for kk, vv in v.items() if kk != "_c"} for k, v in per.items()},
           "top_words": lg["_c"].most_common(40)}
    raw = "".join(r.get("raw_lgg") or "" for r in rows)
    res["apostrophe_variants_raw"] = {f"U+{ord(k):04X} {unicodedata.name(k, '?')}": v
                                      for k, v in Counter(ch for ch in raw if ch in APOS_CHARS).items()}
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
    res["overlap_top2000"] = {f"lugbara~{k}": round(len(top("lugbara") & top(k)) / 2000, 3)
                              for k in langs if k != "lugbara"}
    # how each thesis model's tokenizer cuts Lugbara (thesis 3.2.2: keep the apostrophe inside the word)
    res["tokenizers"] = {}
    apos_words = [w for w, _ in lg["_c"].most_common() if "'" in w][:500]
    for key in ("xlmr", "mt5"):
        try:
            tok = AutoTokenizer.from_pretrained(MODELS[key]["base"])
        except Exception as e:  # offline smoke runs
            print("tokenizer", key, "skipped:", e)
            continue
        fit = {}
        for lang in langs:
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
    for lang in langs:
        cnt = Counter(min(40, len(words(r[lang]))) for r in rows)
        res["length_hist"][lang] = {str(n): cnt.get(n, 0) for n in range(1, 41)}
    if dataset == "local" and (DATA / "train.jsonl").exists():
        res["vs_salt"] = compare_corpora()
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


def compare_corpora():
    """The local corpus next to the SALT sample: size, length, vocabulary, shared sentences, and
    how many test words each training set never contains (explains the cross-test scores)."""
    sets = {name: {s: jl_read(data_dir(name) / f"{s}.jsonl") for s in ("train", "val", "test")}
            for name in ("salt", "local")}
    every = {name: [r for part in v.values() for r in part] for name, v in sets.items()}
    vocab = lambda rs, k: Counter(w for r in rs for w in words(r[k]))

    def unseen(train_rs, test_rs, k):
        v = vocab(train_rs, k)
        toks = [w for r in test_rs for w in words(r[k])]
        return round(sum(1 for w in toks if w not in v) / max(1, len(toks)), 4)

    eng = {name: {r["english"] for r in rs} for name, rs in every.items()}
    eng_train = {name: {r["english"] for r in sets[name]["train"]} for name in sets}
    lgv = {name: vocab(rs, "lugbara") for name, rs in every.items()}
    out = {"sentences": {n: len(rs) for n, rs in every.items()},
           "test_sentences": {n: len(sets[n]["test"]) for n in sets},
           "mean_len": {n: {k: round(sum(len(words(r[k])) for r in rs) / max(1, len(rs)), 2)
                            for k in ("english", "lugbara")} for n, rs in every.items()},
           "distinct_lugbara_words": {"salt": len(lgv["salt"]), "local": len(lgv["local"]),
                                      "shared": len(set(lgv["salt"]) & set(lgv["local"]))},
           "apostrophe_word_share": {n: round(sum(v for w, v in c.items() if "'" in w) / max(1, sum(c.values())), 3)
                                     for n, c in lgv.items()},
           "same_english_sentence": {
               "local_in_salt": sum(1 for r in every["local"] if r["english"] in eng["salt"]),
               "local_test_in_salt_train": sum(1 for r in sets["local"]["test"] if r["english"] in eng_train["salt"]),
               "salt_test_in_local_train": sum(1 for r in sets["salt"]["test"] if r["english"] in eng_train["local"]),
               # within each corpus: test sentences that also occur in its own training split
               "local_test_in_local_train": sum(1 for r in sets["local"]["test"] if r["english"] in eng_train["local"]),
               "salt_test_in_salt_train": sum(1 for r in sets["salt"]["test"] if r["english"] in eng_train["salt"])},
           "unseen_test_words": {k: {f"{a}>{b}": unseen(sets[a]["train"], sets[b]["test"], k)
                                     for a in ("salt", "local") for b in ("salt", "local")}
                                 for k in ("lugbara", "english")}}
    print("SALT vs local:", json.dumps(out))
    return out


def md_table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_analysis(r):
    c, s = r["corpus"], r["stats"]
    local = r.get("dataset") == "local"
    langs = list(c)
    others = [l for l in langs if l != "lugbara"]
    figs = set(r.get("figures") or [])
    fig = lambda name, alt: [f"![{alt}](figures/{name})", ""] if name in figs else []
    n = sum(s["split_sizes"].values())
    drop = " / ".join(f"{s['dropped'].get(k, 0):,}" for k in ("empty", "duplicate", "malformed"))
    split = " / ".join(str(s["split_sizes"][k]) for k in ("train", "val", "test"))
    if local:
        intro = (f"Measured on the {n:,} English–Lugbara pairs of the local corpus ([Akuma, 2025]({LOCAL_SOURCE})) "
                 "used to train these models, after cleaning. Produced by `lugbara_mt.py analyze --dataset local`.")
        sample = [["Rows in the CSV", s["rows"]], ["Rows with Lugbara text", s["nonempty"]["lugbara"]],
                  ["Pairs after cleaning (all used)", s["clean_rows"]],
                  ["Removed (empty / duplicate / malformed)", drop],
                  ["Train / validation / test (seed 13)", split]]
    else:
        intro = ("Measured on the 15,524 SALT sentences used to train these models (the thesis-sized sample "
                 "described in the model card). Produced by `lugbara_mt.py analyze`.")
        sample = [["SALT rows (all splits)", s["salt_rows"]], ["Rows with Lugbara text", s["salt_nonempty"]["lugbara"]],
                  ["Rows with English, Lugbara and Luganda, after cleaning", s["clean_rows"]],
                  ["Removed (empty / duplicate / malformed)", drop],
                  ["Sampled (seed 13)", n], ["Train / validation / test", split]]
    L = ["# Lugbara language analysis", "", intro, "",
         "## Background", "",
         "Lugbara (ISO 639-3 `lgg`, Glottolog `lugb1240`) is a Central Sudanic language of the West "
         "Nile region of Uganda and the neighbouring north-east of the DR Congo. It is unrelated to "
         "Luganda and the other Bantu languages of Uganda. Lugbara is tonal, but tone is not written "
         "in everyday spelling. The apostrophe writes a glottal stop and forms the implosive and "
         "pre-glottalised consonants `'b`, `'d`, `'y` and `'w` (Akuma, 2025, §2.3.3).", "",
         "## The corpus" if local else "## The sample", "",
         md_table(sample, ["", "Sentences"]), "",
         "## Sentence length and vocabulary", "",
         md_table([[k.title(), v["sentences"], v["words"], v["distinct"], v["mean_len"], v["ttr"], v["hapax_share"], v["vowel_final"]]
                   for k, v in c.items()],
                  ["Language", "Sentences", "Words", "Distinct words", "Words per sentence", "Type/token",
                   "Share of words seen once", "Word tokens ending in a vowel"]), "",
         f"The same sentences take **{c['lugbara']['mean_len']} words** in Lugbara, "
         + " and ".join(f"{c[l]['mean_len']} in {l.title()}" for l in others) + ".", "",
         *fig("data_sentence_length.png", "Words per sentence"),
         "### Most frequent Lugbara words", "",
         *fig("data_top_lugbara_words.png", "Most frequent Lugbara words"),
         ", ".join(f"`{w}` ({k})" for w, k in r["top_words"]), "",
         "## Spelling", "",
         *fig("data_apostrophe_letters.png", "Letters following the Lugbara apostrophe"),
         f"* Word tokens containing an apostrophe: **{100 * r['apostrophe_word_share']:.1f}%**",
         "* Apostrophe followed by: " + ", ".join(f"`'{k}` {v}" for k, v in r["apostrophe_followed_by"]),
         f"* Apostrophe characters in the raw {'corpus' if local else 'SALT'} text (all folded to `'`): "
         + (", ".join(f"{k}: {v}" for k, v in r["apostrophe_variants_raw"].items()) or "none"),
         "* Letters outside ASCII: " + (", ".join(f"`{k}` {v}" for k, v in r["non_ascii_letters"].items()) or "none"),
         "* Most common word-initial consonant clusters: " + ", ".join(f"`{k}` {v}" for k, v in r["initial_clusters"]), "",
         "## Relationship to " + " and ".join(l.title() for l in others), "",
         md_table([[k.replace("~", " vs ").title(), f"{100 * v:.1f}%"] for k, v in r["overlap_top2000"].items()],
                  ["Pair", "Shared top-2,000 words"]), "",
         "## How the thesis models' tokenizers cut Lugbara", ""]
    if r["tokenizers"]:
        L += fig("data_tokenizer_pieces.png", "Sub-word pieces per word for each tokenizer")
        L += [md_table([["XLM-R" if k == "xlmr" else "mT5", *[v["pieces_per_word"].get(l, "") for l in langs],
                         f"{100 * v['apostrophe_cut_off_share']:.1f}%"] for k, v in r["tokenizers"].items()],
                       ["Tokenizer", *[f"Pieces/word {l.title()}" for l in langs],
                        "Apostrophe words where `'` becomes its own piece"]), "",
              "The thesis kept the apostrophe inside the word when tokenising (§3.2.2). The last column "
              "shows how often each pre-trained tokenizer instead cuts it off, which is one reason the "
              "model card also reports apostrophe integrity of the output.", ""]
    v = r.get("vs_salt")
    if v:
        u = v["unseen_test_words"]
        pct = lambda x: f"{100 * x:.1f}%"
        same = v["same_english_sentence"]
        L += ["## Compared with the SALT sample", "",
              md_table([["Sentence pairs", f"{v['sentences']['salt']:,}", f"{v['sentences']['local']:,}"],
                        ["Words per English sentence", v["mean_len"]["salt"]["english"], v["mean_len"]["local"]["english"]],
                        ["Words per Lugbara sentence", v["mean_len"]["salt"]["lugbara"], v["mean_len"]["local"]["lugbara"]],
                        ["Distinct Lugbara words", f"{v['distinct_lugbara_words']['salt']:,}",
                         f"{v['distinct_lugbara_words']['local']:,}"],
                        ["Lugbara word tokens with an apostrophe", pct(v["apostrophe_word_share"]["salt"]),
                         pct(v["apostrophe_word_share"]["local"])]],
                       ["", "SALT sample", "Local corpus"]), "",
              f"* Distinct Lugbara words found in both: **{v['distinct_lugbara_words']['shared']:,}**.",
              f"* Local sentences whose English also appears in the SALT sample: {same['local_in_salt']:,} "
              f"(local test sentences in SALT train: {same['local_test_in_salt_train']:,}; "
              f"SALT test sentences in local train: {same['salt_test_in_local_train']:,}).",
              f"* Test sentences whose English also occurs in the same corpus's training split: local "
              f"{same['local_test_in_local_train']:,} of {v['test_sentences']['local']:,}, "
              f"SALT {same['salt_test_in_salt_train']:,} of {v['test_sentences']['salt']:,}.", "",
              "### Test words a training set never contains", "",
              *fig("data_unseen_words.png", "Share of test words never seen in training"),
              md_table([[("SALT" if a == "salt" else "Local") + " train",
                         *[f"{pct(u['lugbara'][f'{a}>{b}'])} / {pct(u['english'][f'{a}>{b}'])}" for b in ("salt", "local")]]
                        for a in ("salt", "local")],
                       ["Lugbara / English word tokens unseen in …", "SALT test", "Local test"]), "",
              "A model cannot be expected to produce a Lugbara word it never saw in training, so the "
              "off-diagonal cells bound how well a model trained on one corpus can do on the other.", ""]
    L += ["## Limits", "",
          ("* The corpus has no domain labels, so the split is random rather than stratified by domain."
           if local else
           "* SALT sentences are short and mostly about agriculture, health and community life; the "
           "thesis corpus also covered tourism and religion."),
          "* Tone is not written, so words that differ only in tone look the same.",
          "* All text is lowercased, as in the thesis.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- card / push / copy
def card(args):
    cfg = MODELS[args.model]
    local = cfg["dataset"] == "local"
    scores = json.loads((OUT / args.model / "scores.json").read_text())
    tlog = json.loads((OUT / args.model / "train_log.json").read_text())
    gpath = OUT / args.model / "grid.json"
    grid_res = json.loads(gpath.read_text())["results"] if gpath.exists() else []
    stats = json.loads((data_dir(cfg["dataset"]) / "stats.json").read_text())
    repo = args.repo
    langs = sorted({ISO[l] for d in cfg["directions"] for l in d})
    ex_src, ex_tgt = cfg["directions"][0]
    test_name = "local corpus" if local else "SALT"
    rows = []
    for d, v in scores.items():
        ref = THESIS.get(d, {})
        best_ref = max(ref.items(), key=lambda kv: kv[1]) if ref else None
        rows.append([d.replace("-", " → ").title(), v["n"], v["bleu"], v["ter"], v["word_f1"], v["chrf"],
                     v.get("apostrophe_integrity", "–"),
                     f"{best_ref[1]} ({best_ref[0]})" if best_ref else "–"])
    first_test = jl_read(OUT / args.model / f"test_outputs.{ex_src}-{ex_tgt}.jsonl")[:8]
    usage = "\n".join(f"print(translate({r['src'].split(' ', 1)[1]!r}, {r['src'].split(' ', 1)[0]!r}))"
                      for r in first_test[:2])
    notes = "".join(f"* {n}\n" for n in cfg.get("notes", []))
    figdir = OUT / args.model / "figures"
    figs_md = "".join(f"![{alt}](figures/{name})\n\n" for name, alt in (
        ("test_scores.png", "Test scores"), ("quality_by_length.png", "Quality by sentence length"),
        ("training.png", "Training and validation loss"), ("grid_search.png", "Grid search"))
        if (figdir / name).exists())
    model_kind = "XLM-R was the thesis's optimised English→Lugbara model" if cfg["kind"] == "encdec" else \
                 "mT5 was the thesis's optimised Lugbara→English model"
    drop = " / ".join(f"{stats['dropped'].get(k, 0):,}" for k in ("empty", "duplicate", "malformed"))
    sz = stats["split_sizes"]
    grid_str = ", ".join(f"{g['lr']:g}/{g['batch_size']} → {g['val_bleu']} BLEU" for g in grid_res) or "not run"
    if local:
        cp_link = hub_link(cfg["counterpart"])
        about = (f"\nIt is trained on the **local English–Lugbara corpus** published with the thesis "
                 f"([Akuma1992/NLP-data-set]({LOCAL_SOURCE})) instead of SALT, with the same settings as its "
                 f"SALT-trained twin {cp_link}, so the two can be compared.\n")
        data_md = (f"* **Data:** the local English–Lugbara corpus ([Akuma1992/NLP-data-set]({LOCAL_SOURCE}), "
                   f"`datasetnlp .csv`, {stats['rows']:,} rows). Empty / duplicate / malformed pairs removed "
                   f"({drop}), leaving {stats['clean_rows']:,} pairs, split 70/15/15 at random (seed {SEED}): "
                   f"{sz['train']:,} train, {sz['val']:,} validation, {sz['test']:,} test. The corpus has no "
                   "domain labels, so the split is random rather than stratified by domain.")
        hyper_md = (f"* **Hyperparameters:** no new grid search: the learning rate and batch size that the SALT "
                    f"grid search chose for {cp_link} (lr {tlog['lr']:g}, batch {tlog['batch_size']}), optimiser "
                    f"{cfg['optim']}, early stopping on validation loss (patience {cfg.get('patience', 3)}); "
                    f"{tlog['global_step']:,} steps, {tlog['minutes']} minutes on a Kaggle GPU.")
        thesis_note = ("\\*The thesis scores come from the thesis's own split and cleaning of its corpus, so "
                       "they are a reference point rather than an exact like-for-like comparison.")
        limits_md = (f"* Trained on {sz['train']:,} sentence pairs from one corpus; text from other domains "
                     "may be translated less well (see the SALT cross-test above).")
        credits_md = (f"* Data: Charity Driciru Akuma, local English–Lugbara corpus ([GitHub]({LOCAL_SOURCE})). "
                      "Its repository states no licence; ask the author before reusing the data.\n"
                      "* Cross-test data: Sunbird AI, SALT (CC BY-SA 4.0).")
    else:
        about = ""
        data_md = (f"* **Data:** the thesis corpus (15,524 English–Lugbara pairs) is not public, so the same number\n"
                   f"  of sentences was drawn at random (seed {SEED}) from the cleaned [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt)\n"
                   f"  `text-all` table and split 70/15/15 like the thesis: {sz['train']:,} train,\n"
                   f"  {sz['val']:,} validation, {sz['test']:,} test. Luganda uses\n"
                   "  the same sentences. SALT has no domain labels, so the split is random rather than stratified by domain.")
        hyper_md = ("* **Hyperparameters (thesis §3.4.3):** grid search over learning rate × batch size\n"
                    f"  ({grid_str}),\n"
                    f"  then fine-tuning with lr {tlog['lr']:g}, batch {tlog['batch_size']}, optimiser {cfg['optim']},\n"
                    f"  early stopping on validation loss (patience {cfg.get('patience', 3)}); {tlog['global_step']:,} steps,\n"
                    f"  {tlog['minutes']} minutes on a Kaggle GPU.")
        thesis_note = ("\\*The thesis scores were measured on the thesis's own corpus, not on SALT, so they are a\n"
                       "reference point rather than a like-for-like comparison.")
        limits_md = "* Trained on short SALT sentences (agriculture, health, community life); other domains are weaker."
        credits_md = "* Data: Sunbird AI, SALT (CC BY-SA 4.0)."
    cross_md = cross_section_md(OUT / args.model, cfg["counterpart"]) if local else ""
    datasets_line = "" if local else "datasets: [Sunbird/salt]\n"
    md = f"""---
language: [{", ".join(langs)}]
license: {cfg['license']}
library_name: transformers
pipeline_tag: translation
base_model: {cfg['base']}
{datasets_line}metrics: [bleu, ter, chrf]
tags: [translation, lugbara, {"" if local else "luganda, "}low-resource, uganda, muni-university]
---

# {cfg['title']}

Fine-tuned `{cfg['base']}` for {", ".join(f"{a.title()} → {b.title()}" for a, b in cfg['directions'])}.
It follows the model choice and training method of **C. D. Akuma, *An Optimized Machine
Translation Model for Low-Resource Lugbara Language*, MSc Computer Science, Muni University
(October 2025)**: {model_kind}.
{about}
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

## Results ({test_name} test split, {sum(v["n"] for v in scores.values())} sentence pairs)

{md_table(rows, ["Direction", "Sentences", "BLEU ↑", "TER ↓", "Word F1 ↑", "chrF ↑", "Apostrophe integrity ↑", "Thesis best BLEU*"])}

BLEU and TER are sacreBLEU corpus scores; word F1 is unigram precision/recall F1; apostrophe
integrity is the share of apostrophe words in the reference (`'ba`, `a'dule` …) reproduced intact.
{thesis_note}

{figs_md}{cross_md}
### Sample test translations ({ex_src.title()} → {ex_tgt.title()})

{md_table([[r["src"].split(" ", 1)[1], r["ref"], r["hyp"]] for r in first_test], ["Source", "Reference", "Model"])}

## Training

{data_md}
* **Preprocessing (thesis §3.2.2):** lowercased, punctuation normalised, all apostrophe variants
  folded to `'`, empty / duplicate / malformed pairs removed.
* **Model:** {"encoder-decoder warm-started from XLM-R base (Rothe et al., 2020), encoder and decoder weights shared" if cfg["kind"] == "encdec" else "mT5 base, sequence-to-sequence"}.
{hyper_md}

{notes}
See [LUGBARA_ANALYSIS.md](LUGBARA_ANALYSIS.md) for the Lugbara language analysis of the training data.

## Limits

{limits_md}
* Lowercase only. Tone is not written in Lugbara, so tone-only differences are lost.
* Automatic scores use one reference per sentence; Lugbara speakers should review output before use.

## Credits

* Thesis and method: Charity Driciru Akuma, Muni University (2025).
{credits_md}
* Base model: `{cfg['base']}`.
* Earlier work: [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}) (Mountains of the Moon University).
"""
    (OUT / args.model / "model" / "README.md").write_text(md, encoding="utf-8")
    print("wrote card for", repo)


def plots(args):
    """Grid, training, test-score, length and (local runs) SALT-vs-local figures for one model."""
    import lugbara_plots
    cfg = MODELS[args.model]
    name = cfg["title"].split(" ")[0]
    note = ("test sentence pairs from the local corpus. Thesis scores come from the thesis's own split "
            "and cleaning of its corpus." if cfg["dataset"] == "local" else None)
    paths = lugbara_plots.model_figures(OUT / args.model, name, THESIS, test_note=note)
    comp = OUT / args.model / "comparison.json"
    if comp.exists():
        paths.append(lugbara_plots.dataset_figure(json.loads(comp.read_text()),
                                                  OUT / args.model / "figures" / "salt_vs_local.png", name))
    for p in paths:
        print("figure", p)


def copy_figures_and_outputs(model_out, folder):
    """Data figures + this model's figures -> figures/, test outputs -> test_outputs/,
    cross-tests (local runs) -> cross_tests/NAME/."""
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    for src in (OUT / "figures", model_out / "figures"):
        for p in src.glob("*.png") if src.exists() else []:
            shutil.copy(p, folder / "figures" / p.name)
    (folder / "test_outputs").mkdir(exist_ok=True)
    for p in model_out.glob("test_outputs.*.jsonl"):
        shutil.copy(p, folder / "test_outputs" / p.name)
    cross = model_out / "cross"
    for sub in sorted(cross.iterdir()) if cross.exists() else []:
        if sub.is_dir():
            dest = folder / "cross_tests" / sub.name
            dest.mkdir(parents=True, exist_ok=True)
            for p in sub.glob("*.json*"):
                shutil.copy(p, dest / p.name)


def fetch(args):
    """Download a published run's results (no model weights) into out/MODEL, so the comparison and the
    card can be redone without retraining (e.g. after adding the unshared-sentence table)."""
    from huggingface_hub import snapshot_download
    d = OUT / args.model
    snap = Path(snapshot_download(args.repo, local_dir=str(WORK / "fetch" / args.model), allow_patterns=[
        "scores.json", "train_log.json", "grid.json", "comparison.json", "lugbara_analysis.json",
        "LUGBARA_ANALYSIS.md", "test_outputs/*", "cross_tests/*/*", "figures/*"]))
    (d / "model").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    for f in ("scores.json", "train_log.json", "grid.json", "comparison.json"):
        if (snap / f).exists():
            shutil.copy(snap / f, d / f)
    for f in ("lugbara_analysis.json", "LUGBARA_ANALYSIS.md"):
        if (snap / f).exists():
            shutil.copy(snap / f, OUT / f)
    for p in (snap / "test_outputs").glob("*.jsonl"):
        shutil.copy(p, d / p.name)
    for sub in (snap / "cross_tests").glob("*"):
        if sub.is_dir():
            (d / "cross" / sub.name).mkdir(parents=True, exist_ok=True)
            for p in sub.iterdir():
                shutil.copy(p, d / "cross" / sub.name / p.name)
    for p in (snap / "figures").glob("*.png"):
        shutil.copy(p, d / "figures" / p.name)
    print("fetched", args.repo, "->", d)


def push(args):
    from huggingface_hub import HfApi
    api = HfApi()
    cfg = MODELS[args.model]
    if getattr(args, "meta_only", False):
        # card, comparison, charts and test outputs only; the model weights stay as they are
        folder = OUT / args.model / "meta"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        shutil.copy(OUT / args.model / "model" / "README.md", folder / "README.md")
        if (OUT / args.model / "comparison.json").exists():
            shutil.copy(OUT / args.model / "comparison.json", folder / "comparison.json")
        copy_figures_and_outputs(OUT / args.model, folder)
        api.upload_folder(folder_path=str(folder), repo_id=args.repo,
                          commit_message="SALT vs local cross-test also on the sentences the two corpora do not share")
        print("updated card and results of", f"https://huggingface.co/{args.repo}")
        return
    folder = OUT / args.model / "model"
    for f in ("LUGBARA_ANALYSIS.md", "lugbara_analysis.json"):
        if (OUT / f).exists():
            shutil.copy(OUT / f, folder / f)
    for f in ("scores.json", "train_log.json", "grid.json", "comparison.json"):
        if (OUT / args.model / f).exists():
            shutil.copy(OUT / args.model / f, folder / f)
    copy_figures_and_outputs(OUT / args.model, folder)
    api.create_repo(args.repo, exist_ok=True, private=False)
    data = "the local English-Lugbara corpus" if cfg["dataset"] == "local" else "SALT"
    # replace whatever was there before (e.g. the Rutooro copy), keep .gitattributes
    api.upload_folder(folder_path=str(folder), repo_id=args.repo,
                      commit_message=f"{cfg['title']} trained on {data} (thesis method)",
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
def trim_local_for_smoke(n=40):
    """Smoke runs only: keep n validation/test rows of the local split so CPU evaluation is quick."""
    for split in ("val", "test"):
        p = DATA_LOCAL / f"{split}.jsonl"
        jl_write(p, jl_read(p)[:n])


def smoke(args):
    """Tiny end-to-end run on CPU: prepare, grid (1 point), train, evaluate, analyze, plots, card.
    SALT rows are synthetic; a *_local model also uses the real local corpus (downloaded), runs both
    cross-tests (its own model stands in for the SALT counterpart) and the 2 x 2 comparison.
    --base-root holds local copies of the base models (xlm-roberta-base/, mt5-small/)."""
    if args.base_root:
        for c in MODELS.values():
            c["base"] = str(args.base_root / ("xlm-roberta-base" if c["kind"] == "encdec" else "mt5-small"))
    cfg = MODELS[args.model]
    local = cfg["dataset"] == "local"
    cfg["optim"] = "adafactor"
    if local:
        cfg["bs"] = 2
    else:
        cfg["grid_lr"], cfg["grid_bs"] = cfg["grid_lr"][:1], [2]
    ns = argparse.Namespace(fake=40, model=args.model, base=None, probe=4, grid_epochs=1, max_steps=3,
                            epochs=1, no_grid=local, limit=4, repo="local/smoke", dataset="salt",
                            on=None, counterpart=False, from_repo=None, revision=None, name=None)
    with_ = lambda **kw: argparse.Namespace(**{**vars(ns), **kw})
    prepare(ns)
    if local:
        prepare(with_(dataset="local"))
        trim_local_for_smoke()
    grid(ns)
    train(ns)
    evaluate(ns)
    if local:
        evaluate(with_(on="salt", name="on_salt"))
        evaluate(with_(on="local", from_repo=str(OUT / args.model / "model"), name="salt_model_on_local"))
        compare(ns)
    analyze(with_(dataset=cfg["dataset"]))
    plots(ns)
    card(ns)
    # reload check
    tok, model = build(args.model, from_dir=OUT / args.model / "model")
    print("reload OK:", translate(tok, model, [prefix(cfg["directions"][0][1]) + "a 'ba ri"], beams=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "grid", "train", "evaluate", "compare", "analyze", "plots", "card",
                                    "fetch", "push", "copy", "smoke"])
    ap.add_argument("--model", choices=list(MODELS), default="xlmr")
    ap.add_argument("--dataset", choices=["salt", "local"], default=None,
                    help="prepare/analyze: SALT sample (default) or the local corpus")
    ap.add_argument("--base", help="local path or hub id overriding the base model")
    ap.add_argument("--base-root", type=Path, help="smoke: folder holding xlm-roberta-base/ and mt5-small/")
    ap.add_argument("--repo")
    ap.add_argument("--fake", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--grid-epochs", type=float, default=None, help="default: MODELS[model]['grid_epochs']")
    ap.add_argument("--probe", type=int, default=300, help="validation sentences scored per grid point")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--on", choices=["salt", "local"], help="evaluate: test split (default: the model's dataset)")
    ap.add_argument("--counterpart", action="store_true", help="evaluate: the published SALT-trained twin")
    ap.add_argument("--from-repo", help="evaluate: any published model (hub id or folder)")
    ap.add_argument("--revision", help="evaluate: revision of --from-repo")
    ap.add_argument("--name", help="evaluate: write to out/MODEL/cross/NAME/")
    ap.add_argument("--meta-only", action="store_true", help="push: card, comparison and charts only (no weights)")
    args = ap.parse_args()
    random.seed(SEED)
    {"prepare": prepare, "grid": grid, "train": train, "evaluate": evaluate, "compare": compare,
     "analyze": analyze, "plots": plots, "card": card, "fetch": fetch, "push": push, "copy": copy,
     "smoke": smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
