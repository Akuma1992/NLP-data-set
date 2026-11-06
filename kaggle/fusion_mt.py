"""XLM-R + mT5 fusion translator: an ensemble of the two models of Akuma (2025).

The thesis found XLM-R best for English->Lugbara and mT5 best for Lugbara->English. This model
combines them in one network instead of choosing per direction:

  source ──► XLM-R encoder ──► projection + LayerNorm ─┐
        └──► mT5 encoder ──────────────────────────────┴─► concatenated memory ──► mT5 decoder ──► target

* XLM-R contributes its cross-lingual sentence understanding; mT5 contributes its encoder and its
  generative decoder. At every output word the decoder attends over BOTH encoders' states at once
  (dual-encoder fusion; cf. Zhu et al., 2020, "Incorporating BERT into Neural Machine Translation").
* A learned source-type vector marks which encoder each state came from.
* One model, four directions (English<->Lugbara, English<->Luganda), selected by the same
  target tags as the single models (<2lgg>, <2lug>, <2en>).
* Each part learns at the rate its own grid search chose: mT5 + fusion layers 1e-3 (mT5's best),
  XLM-R 3e-5 (XLM-R's best). XLM-R's 192M-parameter word-embedding table is frozen to fit a T4.

Data, split, metrics and analysis come from lugbara_mt.py (same seed, same 15,524 sentences),
so the scores compare directly with the single models.

Subcommands:  train | evaluate | compare | plots | card | push | smoke   [--dataset salt|local]
--dataset local trains the same network English <-> Lugbara on the local corpus of Akuma (2025)
and cross-tests it against the SALT-trained fusion model (2 x 2, like lugbara_mt.py compare).
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import lugbara_mt as L

FUSION = {
    "mt5": "google/mt5-base",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "directions": [("english", "lugbara"), ("english", "luganda"),
                   ("lugbara", "english"), ("luganda", "english")],
    "lr": 1e-3,          # mT5 grid winner (lugbara-english-translator)
    "xlmr_lr": 3e-5,     # XLM-R grid winner (lugbara-multilingual-translator)
    "bs": 16, "epochs": 15, "patience": 3,
    "title": "XLM-R + mT5 fusion translator (English ↔ Lugbara, English ↔ Luganda)",
    "license": "apache-2.0",
    "compare": {"XLM-R alone": "ckuma/lugbara-multilingual-translator",
                "mT5 alone": "ckuma/lugbara-english-translator"},
}
# --dataset local: the same network and settings, English <-> Lugbara, trained on the local corpus
# of Akuma (2025) and cross-tested against the SALT-trained fusion model
LOCAL = {
    "directions": [("english", "lugbara"), ("lugbara", "english")],
    "title": "XLM-R + mT5 fusion translator (English ↔ Lugbara, local corpus)",
    "compare": {"XLM-R alone (local)": "ckuma/lugbara-multilingual-translator-local",
                "mT5 alone (local)": "ckuma/lugbara-english-translator-local"},
    "counterpart": {"repo": "ckuma/lugbara-ensemble-translator", "revision": None},
}
DATASET = "salt"
OUTF = L.OUT / "fusion"


def configure(dataset):
    """Point every subcommand at one corpus (salt: four directions; local: English <-> Lugbara)."""
    global DATASET, OUTF
    DATASET = dataset or "salt"
    if DATASET == "local":
        FUSION.update(LOCAL)
        OUTF = L.OUT / "fusion_local"


# ---------------------------------------------------------------- model
def _torch():
    import torch
    from torch import nn
    return torch, nn


def make_model_class():
    torch, nn = _torch()
    from transformers.modeling_outputs import BaseModelOutput

    class FusionTranslator(nn.Module):
        def __init__(self, mt5, xlmr):
            super().__init__()
            self.mt5, self.xlmr = mt5, xlmr
            d_x, d_m = xlmr.config.hidden_size, mt5.config.d_model
            self.proj = nn.Linear(d_x, d_m)
            self.norm = nn.LayerNorm(d_m)
            self.src_type = nn.Parameter(torch.zeros(2, d_m))   # 0 = mT5 states, 1 = XLM-R states

        def encode(self, input_ids, attention_mask, x_input_ids, x_attention_mask):
            hm = self.mt5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            hx = self.xlmr(input_ids=x_input_ids, attention_mask=x_attention_mask).last_hidden_state
            hx = self.norm(self.proj(hx))
            h = torch.cat([hm + self.src_type[0], hx + self.src_type[1]], dim=1)
            mask = torch.cat([attention_mask, x_attention_mask], dim=1)
            return BaseModelOutput(last_hidden_state=h), mask

        def forward(self, input_ids, attention_mask, x_input_ids, x_attention_mask, labels=None):
            enc, mask = self.encode(input_ids, attention_mask, x_input_ids, x_attention_mask)
            return self.mt5(encoder_outputs=enc, attention_mask=mask, labels=labels)

        @torch.no_grad()
        def generate(self, input_ids, attention_mask, x_input_ids, x_attention_mask, **gen):
            enc, mask = self.encode(input_ids, attention_mask, x_input_ids, x_attention_mask)
            return self.mt5.generate(encoder_outputs=enc, attention_mask=mask, **gen)

    return FusionTranslator


def build(mt5_name=None, xlmr_name=None, from_dir=None):
    torch, _ = _torch()
    from transformers import AutoModel, AutoTokenizer, MT5ForConditionalGeneration
    cls = make_model_class()
    if from_dir:
        d = Path(from_dir)
        tm, tx = AutoTokenizer.from_pretrained(d / "mt5"), AutoTokenizer.from_pretrained(d / "xlmr")
        model = cls(MT5ForConditionalGeneration.from_pretrained(d / "mt5"),
                    AutoModel.from_pretrained(d / "xlmr", add_pooling_layer=False))
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(d / "fusion_head.safetensors")), strict=False)
        return tm, tx, model
    mt5_name, xlmr_name = mt5_name or FUSION["mt5"], xlmr_name or FUSION["xlmr"]
    tm, tx = AutoTokenizer.from_pretrained(mt5_name), AutoTokenizer.from_pretrained(xlmr_name)
    model = cls(MT5ForConditionalGeneration.from_pretrained(mt5_name),
                AutoModel.from_pretrained(xlmr_name, add_pooling_layer=False))
    model.xlmr.embeddings.word_embeddings.weight.requires_grad_(False)
    g = model.mt5.generation_config
    g.max_length, g.num_beams, g.early_stopping = L.MAX_LEN, 4, True
    return tm, tx, model


def head_state(model):
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
            if k.startswith(("proj.", "norm.", "src_type"))}


def save(model, tm, tx, out):
    from safetensors.torch import save_file
    out = Path(out)
    (out / "mt5").mkdir(parents=True, exist_ok=True)
    model.mt5.save_pretrained(out / "mt5")
    tm.save_pretrained(out / "mt5")
    model.xlmr.save_pretrained(out / "xlmr")
    tx.save_pretrained(out / "xlmr")
    save_file(head_state(model), str(out / "fusion_head.safetensors"))
    (out / "fusion_config.json").write_text(json.dumps(
        {"architecture": "XLM-R + mT5 dual-encoder fusion", "mt5": FUSION["mt5"], "xlmr": FUSION["xlmr"],
         "directions": ["-".join(d) for d in FUSION["directions"]], "trained_on": DATASET,
         "tags": {t: L.prefix(t).strip() for t in L.TAG},
         "lowercase": L.LOWERCASE, "max_len": L.MAX_LEN}, indent=2))
    shutil.copy(Path(__file__), out / "fusion_mt.py")
    shutil.copy(Path(L.__file__), out / "lugbara_mt.py")


# ---------------------------------------------------------------- data
def dataset(rows, tm, tx):
    from datasets import Dataset

    def enc(b):
        m = tm(b["src"], max_length=L.MAX_LEN, truncation=True)
        x = tx(b["src"], max_length=L.MAX_LEN, truncation=True)
        y = tm(text_target=b["tgt"], max_length=L.MAX_LEN, truncation=True)["input_ids"]
        return {"input_ids": m["input_ids"], "attention_mask": m["attention_mask"],
                "x_input_ids": x["input_ids"], "x_attention_mask": x["attention_mask"], "labels": y}
    return Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in rows]).map(
        enc, batched=True, remove_columns=["src", "tgt"])


def collator(tm, tx):
    torch, _ = _torch()

    def pad(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (n - len(s)) for s in seqs])

    def collate(feats):
        return {"input_ids": pad([f["input_ids"] for f in feats], tm.pad_token_id),
                "attention_mask": pad([f["attention_mask"] for f in feats], 0),
                "x_input_ids": pad([f["x_input_ids"] for f in feats], tx.pad_token_id),
                "x_attention_mask": pad([f["x_attention_mask"] for f in feats], 0),
                "labels": pad([f["labels"] for f in feats], -100)}
    return collate


def translate(tm, tx, model, srcs, bs=32, beams=4):
    torch, _ = _torch()
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(srcs), bs):
        chunk = srcs[i:i + bs]
        m = tm(chunk, return_tensors="pt", padding=True, truncation=True, max_length=L.MAX_LEN).to(dev)
        x = tx(chunk, return_tensors="pt", padding=True, truncation=True, max_length=L.MAX_LEN).to(dev)
        y = model.generate(m["input_ids"], m["attention_mask"], x["input_ids"], x["attention_mask"],
                           num_beams=beams, max_new_tokens=L.MAX_LEN)
        out += [t.strip() for t in tm.batch_decode(y, skip_special_tokens=True)]
    return out


# ---------------------------------------------------------------- train / evaluate
def train(args):
    torch, _ = _torch()
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    cpu = not torch.cuda.is_available()
    tm, tx, model = build(args.mt5, args.xlmr)
    dirs = FUSION["directions"]
    tr = dataset(L.pairs("train", dirs, dataset=DATASET), tm, tx)
    va = dataset(L.pairs("val", dirs, dataset=DATASET), tm, tx)
    targs = TrainingArguments(
        output_dir=str(OUTF / "checkpoints"), per_device_train_batch_size=args.bs or FUSION["bs"],
        per_device_eval_batch_size=8, prediction_loss_only=True, learning_rate=FUSION["lr"],
        warmup_ratio=0.05, lr_scheduler_type="linear", weight_decay=0.0,
        num_train_epochs=args.epochs or FUSION["epochs"], max_steps=args.max_steps,
        use_cpu=cpu, fp16=False,                       # mT5 overflows in fp16
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, save_only_model=True,
        save_safetensors=False,                        # mT5 shares its embedding matrix
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=50, report_to="none", seed=L.SEED, remove_unused_columns=False,
        group_by_length=True, dataloader_num_workers=0 if cpu else 2)
    cls = L.group_lr_trainer_class(Trainer)
    cls.lr_mult = {"xlmr.": FUSION["xlmr_lr"] / FUSION["lr"]}
    cls.optimizer_kind = "adafactor"
    trainer = cls(model=model, args=targs, train_dataset=tr, eval_dataset=va, data_collator=collator(tm, tx),
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=FUSION["patience"])])
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"fusion model: {n_all / 1e6:.0f}M parameters, {n_tr / 1e6:.0f}M trainable; "
          f"{len(tr)} training pairs over {len(dirs)} directions", flush=True)
    t0 = time.time()
    trainer.train()
    save(model, tm, tx, OUTF / "model")
    (OUTF / "train_log.json").write_text(json.dumps(
        {"lr": FUSION["lr"], "xlmr_lr": FUSION["xlmr_lr"], "batch_size": targs.per_device_train_batch_size,
         "minutes": round((time.time() - t0) / 60, 1), "global_step": trainer.state.global_step,
         "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
         "params_total": n_all, "params_trainable": n_tr, "history": trainer.state.log_history}, indent=1))
    shutil.rmtree(OUTF / "checkpoints", ignore_errors=True)
    print("saved", OUTF / "model", flush=True)


def evaluate(args):
    """Score on a test split (default: this run's model on its own corpus's test split).
    --on salt|local, --counterpart (the SALT-trained fusion model), --from-repo R, --name N -> cross/N/."""
    torch, _ = _torch()
    on = getattr(args, "on", None) or DATASET
    src_model, rev = getattr(args, "from_repo", None), None
    if getattr(args, "counterpart", False):
        src_model, rev = FUSION["counterpart"]["repo"], FUSION["counterpart"]["revision"]
    if src_model and not Path(src_model).exists():
        from huggingface_hub import snapshot_download
        src_model = snapshot_download(src_model, revision=rev)
    tm, tx, model = build(from_dir=src_model or OUTF / "model")
    if torch.cuda.is_available():
        model = model.cuda()
    name = getattr(args, "name", None)
    dest = OUTF / "cross" / name if name else OUTF
    dest.mkdir(parents=True, exist_ok=True)
    res = {}
    for src, tgt in FUSION["directions"]:
        rows = L.pairs("test", [(src, tgt)], limit=args.limit, dataset=on)
        if not rows:
            continue
        hyps = translate(tm, tx, model, [r["src"] for r in rows])
        d = f"{src}-{tgt}"
        res[d] = L.score(hyps, [r["tgt"] for r in rows], tgt)
        who = "SALT-trained fusion" if getattr(args, "counterpart", False) else "this model"
        print(f"[{who} on {on} test]", d, res[d], flush=True)
        L.jl_write(dest / f"test_outputs.{d}.jsonl",
                   [{"id": r["id"], "src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in zip(rows, hyps)])
    (dest / "scores.json").write_text(json.dumps(res, indent=2))
    if name:
        (dest / "about.json").write_text(json.dumps(
            {"model": FUSION["counterpart"]["repo"] if getattr(args, "counterpart", False)
             else str(getattr(args, "from_repo", None) or "this run (fusion)"), "revision": rev, "test_set": on}, indent=2))


def compare(args):
    """Local runs: trained on SALT vs trained on the local corpus x tested on both (2 x 2)."""
    return L.cross_table(OUTF, FUSION["counterpart"], FUSION["directions"], "Fusion")


# ---------------------------------------------------------------- plots / card / push
def plots(args):
    """Training, test-score and length figures, the comparison with the single models and,
    for a local run, trained on SALT vs trained on the local corpus."""
    import lugbara_plots as P
    local = DATASET == "local"
    note = ("test sentence pairs from the local corpus. Thesis scores come from the thesis's own split "
            "and cleaning of its corpus." if local else None)
    paths = P.model_figures(OUTF, "Fusion", L.THESIS, test_note=note)
    comp = compare_scores()
    comp["XLM-R + mT5 fusion"] = json.loads((OUTF / "scores.json").read_text())
    for metric in ("bleu", "chrf"):
        paths.append(P.compare_figure(comp, OUTF / "figures" / f"compare_{metric}.png", metric,
                                      f"Test {'BLEU' if metric == 'bleu' else 'chrF'}: single models vs the fusion",
                                      note="Same local-corpus test split for every model." if local else None))
    if (OUTF / "comparison.json").exists():
        paths.append(P.dataset_figure(json.loads((OUTF / "comparison.json").read_text()),
                                      OUTF / "figures" / "salt_vs_local.png", "Fusion"))
    for p in paths:
        print("figure", p)


def compare_scores():
    """Scores of the single models, read from their Hub repos (same test split)."""
    out = {}
    try:
        from huggingface_hub import hf_hub_download
        for name, repo in FUSION["compare"].items():
            try:
                out[name] = json.loads(Path(hf_hub_download(repo, "scores.json")).read_text())
            except Exception as e:  # repo not published yet
                print("compare: skip", repo, type(e).__name__)
    except ImportError:
        pass
    return out


def card(args):
    local = DATASET == "local"
    scores = json.loads((OUTF / "scores.json").read_text())
    tlog = json.loads((OUTF / "train_log.json").read_text())
    stats = json.loads((L.data_dir(DATASET) / "stats.json").read_text())
    sz = stats["split_sizes"]
    comp = compare_scores()
    rows = []
    for d, v in scores.items():
        singles = " / ".join(f"{c[d]['bleu']} ({n})" for n, c in comp.items() if d in c) or "–"
        thesis = L.THESIS.get(d, {})
        tb = max(thesis.items(), key=lambda kv: kv[1]) if thesis else None
        rows.append([d.replace("-", " → ").title(), v["n"], v["bleu"], v["ter"], v["word_f1"], v["chrf"],
                     v.get("apostrophe_integrity", "–"), singles, f"{tb[1]} ({tb[0]})" if tb else "–"])
    samples = {}
    for src, tgt in FUSION["directions"]:
        samples[f"{src}-{tgt}"] = L.jl_read(OUTF / f"test_outputs.{src}-{tgt}.jsonl")[:5]
    sample_md = "\n\n".join(
        f"**{d.replace('-', ' → ').title()}**\n\n" + L.md_table(
            [[r["src"].split(" ", 1)[1], r["ref"], r["hyp"]] for r in rs], ["Source", "Reference", "Fusion model"])
        for d, rs in samples.items())
    repo = args.repo
    figdir = OUTF / "figures"
    figs_md = "".join(f"![{alt}](figures/{name})\n\n" for name, alt in (
        ("compare_bleu.png", "BLEU: single models vs fusion"), ("compare_chrf.png", "chrF: single models vs fusion"),
        ("test_scores.png", "Test scores"), ("quality_by_length.png", "Quality by sentence length"),
        ("training.png", "Training and validation loss")) if (figdir / name).exists())
    if local:
        cp = FUSION["counterpart"]
        langs, ds_line, lug_tag = "[en, lgg]", "", ""
        scope = "both directions between English and Lugbara"
        about = (f"\nIt is trained on the **local English–Lugbara corpus** published with the thesis "
                 f"([Akuma1992/NLP-data-set]({L.LOCAL_SOURCE})) instead of SALT, with the same settings as the "
                 f"SALT-trained fusion model {L.hub_link(cp)}, so the two can be compared.\n")
        tags_md = "Tags: `<2lgg>` → Lugbara, `<2en>` → English. Use lowercase input, as in training."
        test_name = "local corpus test split"
        thesis_note = ("\\*Thesis scores come from the thesis's own split and cleaning of its corpus, so they are "
                       "a reference point only.")
        drop = " / ".join(f"{stats['dropped'].get(k, 0):,}" for k in ("empty", "duplicate", "malformed"))
        data_md = (f"* **Data:** the local English–Lugbara corpus ([Akuma1992/NLP-data-set]({L.LOCAL_SOURCE}), "
                   f"{stats['rows']:,} rows; {drop} empty / duplicate / malformed pairs removed), "
                   f"{sz['train']:,} / {sz['val']:,} / {sz['test']:,} pairs (train / validation / test, random, "
                   f"seed {L.SEED}); every pair is used in both directions.")
        limits_md = (f"* Trained on {sz['train']:,} sentence pairs from one corpus; other domains may be translated "
                     "less well (see the SALT cross-test above).")
        credits_md = (f"Thesis, method and data: Charity Driciru Akuma, Muni University (2025); the corpus "
                      f"repository ([GitHub]({L.LOCAL_SOURCE})) states no licence, so ask the author before reusing "
                      "the data · Cross-test data: Sunbird AI, SALT (CC BY-SA 4.0)")
        cross_md = L.cross_section_md(OUTF, cp, what="fusion model")
    else:
        langs, ds_line, lug_tag = "[en, lg, lgg]", "datasets: [Sunbird/salt]\n", "luganda, "
        scope = "all four directions"
        about = ""
        tags_md = "Tags: `<2lgg>` → Lugbara, `<2lug>` → Luganda, `<2en>` → English. Use lowercase input, as in training."
        test_name = "test split"
        thesis_note = ("\\*Thesis scores were measured on\nthe thesis's own corpus, not on SALT, so they are a "
                       "reference point only.")
        data_md = (f"* **Data:** {sz['train']:,} / {sz['val']:,} / {sz['test']:,}\n"
                   f"  sentences (train / validation / test) drawn at random (seed {L.SEED}) from Sunbird SALT, the\n"
                   "  thesis's 15,524-sentence, 70/15/15 design; every sentence is used in all four directions.")
        limits_md = "* Short SALT sentences (agriculture, health, community life); other domains are weaker."
        credits_md = "Thesis and method: Charity Driciru Akuma, Muni University (2025) · Data: Sunbird AI, SALT (CC BY-SA 4.0)"
        cross_md = ""
    md = f"""---
language: {langs}
license: {FUSION['license']}
library_name: transformers
pipeline_tag: translation
base_model: [{FUSION['mt5']}, {FUSION['xlmr']}]
{ds_line}metrics: [bleu, ter, chrf]
tags: [translation, lugbara, {lug_tag}low-resource, uganda, muni-university, ensemble, fusion]
---

# {FUSION['title']}

An ensemble of the two models of **C. D. Akuma, *An Optimized Machine Translation Model for
Low-Resource Lugbara Language*, MSc Computer Science, Muni University (October 2025)**. The thesis
found XLM-R best for English→Lugbara and mT5 best for Lugbara→English; this model fuses them into
one network that translates in {scope}.
{about}
```
source ──► XLM-R encoder ──► projection + LayerNorm ─┐
      └──► mT5 encoder ──────────────────────────────┴─► joint memory ──► mT5 decoder ──► translation
```

The mT5 decoder attends over the states of **both** encoders at every output word; a learned
source-type vector tells it which encoder each state came from (dual-encoder fusion, cf. Zhu et
al., 2020, *Incorporating BERT into Neural Machine Translation*).

## How to use

The repository holds the two encoders' weights (`mt5/`, `xlmr/`), the fusion layers
(`fusion_head.safetensors`) and the model code (`fusion_mt.py`, `lugbara_mt.py`).

```python
import sys
from huggingface_hub import snapshot_download
d = snapshot_download("{repo}")
sys.path.insert(0, d)
import fusion_mt

tm, tx, model = fusion_mt.build(from_dir=d)
print(fusion_mt.translate(tm, tx, model, ["<2lgg> farmers should plant early when the rains begin."]))
print(fusion_mt.translate(tm, tx, model, ["<2en> " + "<a lugbara sentence, lowercase>"]))
```

{tags_md}

## Results ({test_name}, {sum(v['n'] for v in scores.values())} sentence pairs)

{L.md_table(rows, ["Direction", "Sentences", "BLEU ↑", "TER ↓", "Word F1 ↑", "chrF ↑", "Apostrophe integrity ↑", "BLEU of the single models", "Thesis best BLEU*"])}

The single models were trained and scored on exactly the same split. {thesis_note}

{figs_md}{cross_md}
### Sample test translations

{sample_md}

## Training

{data_md}
* **Learning rates:** mT5 and the fusion layers {FUSION['lr']:g} (mT5's grid-search winner on SALT);
  XLM-R {FUSION['xlmr_lr']:g} (XLM-R's grid-search winner on SALT); Adafactor; XLM-R's word embeddings frozen.
* **Training:** batch {tlog['batch_size']}, early stopping on validation loss (patience {FUSION['patience']}),
  {tlog['global_step']:,} steps, {tlog['minutes']} minutes on a Kaggle T4;
  {tlog['params_total'] / 1e6:.0f}M parameters ({tlog['params_trainable'] / 1e6:.0f}M trained).

See [LUGBARA_ANALYSIS.md](LUGBARA_ANALYSIS.md) for the Lugbara language analysis of the training data.

## Limits

{limits_md}
* Lowercase only; Lugbara tone is not written, so tone-only differences are lost.
* Automatic scores use one reference per sentence; Lugbara speakers should review output before use.

## Credits

{credits_md}
· Base models: `{FUSION['mt5']}`, `{FUSION['xlmr']}` · Earlier work: [{L.SOURCE_REPO}](https://huggingface.co/{L.SOURCE_REPO}).
"""
    (OUTF / "model" / "README.md").write_text(md, encoding="utf-8")
    print("wrote card for", repo)


def push(args):
    from huggingface_hub import HfApi
    folder = OUTF / "model"
    for f in ("LUGBARA_ANALYSIS.md", "lugbara_analysis.json"):
        if (L.OUT / f).exists():
            shutil.copy(L.OUT / f, folder / f)
    for f in ("scores.json", "train_log.json", "comparison.json"):
        if (OUTF / f).exists():
            shutil.copy(OUTF / f, folder / f)
    L.copy_figures_and_outputs(OUTF, folder)
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=False)
    data = "the local English-Lugbara corpus" if DATASET == "local" else "SALT"
    api.upload_folder(folder_path=str(folder), repo_id=args.repo,
                      commit_message=f"XLM-R + mT5 fusion translator trained on {data} (Akuma 2025 models)")
    print("pushed", f"https://huggingface.co/{args.repo}")


# ---------------------------------------------------------------- smoke
def smoke(args):
    """CPU end-to-end check with small local models. SALT rows are synthetic; --dataset local also
    uses the real local corpus and runs the cross-tests (its own model stands in for the SALT one)."""
    FUSION["mt5"] = str(args.base_root / "mt5-small")
    FUSION["xlmr"] = str(args.base_root / "xlm-roberta-base")
    FUSION["compare"] = {}
    for c in L.MODELS.values():   # tokenizers for the analysis
        c["base"] = FUSION["xlmr"] if c["kind"] == "encdec" else FUSION["mt5"]
    ns = lambda **kw: argparse.Namespace(**{"limit": 3, "on": None, "counterpart": False, "from_repo": None,
                                            "name": None, **kw})
    L.prepare(argparse.Namespace(fake=40))
    if DATASET == "local":
        L.prepare(argparse.Namespace(fake=0, dataset="local"))
        L.trim_local_for_smoke()
    train(argparse.Namespace(mt5=None, xlmr=None, bs=2, epochs=1, max_steps=3))
    evaluate(ns())
    if DATASET == "local":
        evaluate(ns(on="salt", name="on_salt"))
        evaluate(ns(on="local", from_repo=str(OUTF / "model"), name="salt_model_on_local"))
        compare(ns())
    L.analyze(argparse.Namespace(dataset=DATASET))
    plots(argparse.Namespace())
    card(argparse.Namespace(repo="local/fusion-smoke"))
    tm, tx, model = build(from_dir=OUTF / "model")
    print("reload OK:", ascii(translate(tm, tx, model, [L.prefix("lugbara") + "the market opens early"], beams=2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate", "compare", "plots", "card", "push", "smoke"])
    ap.add_argument("--dataset", choices=["salt", "local"], default="salt",
                    help="salt: four directions on the SALT sample; local: English <-> Lugbara on the local corpus")
    ap.add_argument("--repo")
    ap.add_argument("--mt5")
    ap.add_argument("--xlmr")
    ap.add_argument("--bs", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base-root", type=Path)
    ap.add_argument("--on", choices=["salt", "local"], help="evaluate: test split (default: the run's corpus)")
    ap.add_argument("--counterpart", action="store_true", help="evaluate: the SALT-trained fusion model")
    ap.add_argument("--from-repo", help="evaluate: another fusion model (hub id or folder)")
    ap.add_argument("--name", help="evaluate: write to out/fusion_local/cross/NAME/")
    args = ap.parse_args()
    configure(args.dataset)
    {"train": train, "evaluate": evaluate, "compare": compare, "plots": plots, "card": card, "push": push,
     "smoke": smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
