"""Build the Kaggle notebooks from the pipeline scripts (run again after editing a script).

01  XLM-R   English -> Lugbara / Luganda          (thesis model, v2)
02  mT5     Lugbara / Luganda -> English          (thesis model)
03  Fusion  XLM-R + mT5 ensemble, all 4 directions
04  Report  plots comparing every published model (CPU only)
05  mT5     Lugbara -> English on the local corpus, cross-tested against the SALT-trained mT5
06  XLM-R   English -> Lugbara on the local corpus, cross-tested against the SALT-trained XLM-R (v1)
07  Fusion  XLM-R + mT5, English <-> Lugbara on the local corpus, cross-tested against the SALT fusion
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = {name: (HERE / name).read_text(encoding="utf-8")
       for name in ("lugbara_mt.py", "lugbara_plots.py", "fusion_mt.py")}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def with_ids(cells):
    """Copies of the cells with unique ids (nbformat 4.5); shared cells appear in several notebooks."""
    return [{**c, "id": f"c{i:03d}"} for i, c in enumerate(cells, 1)]


def setup_md(title, body, repo, hours, gpu=True):
    steps = ["*Accelerator*: **GPU T4 x2** (only one GPU is used, so the batch sizes match the thesis)."] if gpu else \
            ["*Accelerator*: **None** (no GPU needed)."]
    steps.append("*Internet*: **on** (needs a phone-verified Kaggle account).")
    if gpu:
        steps.append("*Add-ons → Secrets*: a secret named **`HF_TOKEN`** holding a Hugging Face **Write** token of "
                     "the account that owns `ckuma` and has accepted the SALT terms, ticked for this notebook. "
                     "Copy the token by selecting it and pressing Ctrl+C (the site's Copy button may not work in "
                     "every browser).")
    return md(f"""
# {title}

{body}

**Before you run it** (right-hand panel → *Settings* and *Add-ons*):
""" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) + f"""

Then **Save Version → Save & Run All (Commit)**. It runs in the background ({hours})
{"and uploads to **[" + repo + "](https://huggingface.co/" + repo + ")** when it finishes; the browser can be closed." if repo else "."}
""")


PIP = code('!pip install -q "transformers>=4.46,<5" "datasets>=2.19" sacrebleu sentencepiece protobuf accelerate matplotlib')
LOGIN = code("""
import re
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami
raw = UserSecretsClient().get_secret("HF_TOKEN") or ""
found = re.findall(r"hf_[A-Za-z0-9]{30,}", raw)
print("HF_TOKEN secret check (the token itself is never printed):", len(raw), "characters; starts with hf_:", raw.strip().startswith("hf_"), "; contains '...':", "..." in raw, "; full-length tokens found:", len(found))
login(token=found[0] if found else raw.strip(), add_to_git_credential=False)
print("Signed in to Hugging Face as", whoami()["name"])
""")
STRICT = [md("Stop the whole run as soon as any step fails (a failing `!command` does not stop a notebook by itself)."),
          code("""
ip = get_ipython()
_system = ip.system
def _strict_system(cmd):
    _system(cmd)
    if ip.user_ns.get("_exit_code", 0):
        raise RuntimeError("step failed: " + cmd)
ip.system = _strict_system
print("steps will stop the run on failure")
""")]
SHOW_DATA = code("""
import lugbara_plots as P
from pathlib import Path
P.show(sorted(Path("out/figures").glob("*.png")))
""")


def writefiles(*names):
    cells = [md("The pipeline scripts (written to disk so every step runs the same code).")]
    for n in names:
        cells.append(code(f"%%writefile {n}\n" + SRC[n]))
    return cells


def data_cells():
    return [md("## 1. Data: sample and split SALT like the thesis, and analyse the Lugbara"),
            code("!python lugbara_mt.py prepare\n!python lugbara_mt.py analyze"),
            md("### Lugbara language analysis: charts"), SHOW_DATA]


def single_model(name, nb):
    cells = [setup_md(nb["title"], nb["body"], nb["repo"], nb["hours"]), PIP, LOGIN,
             code(f'MODEL = "{nb["model"]}"\nREPO = "{nb["repo"]}"\nRUN_GRID = True      # thesis 3.4.3: learning-rate x batch-size grid search'),
             *writefiles("lugbara_mt.py", "lugbara_plots.py"), *STRICT]
    if nb.get("copy_first"):
        cells += [md("Copy the Rutooro model into the repo if it is still empty; the trained model replaces it at the end."),
                  code("!python lugbara_mt.py copy --repo $REPO")]
    cells += data_cells() + [
        md("## 2. Grid search (learning rate × batch size)"),
        code("if RUN_GRID:\n    !python lugbara_mt.py grid --model $MODEL\n!cat out/$MODEL/grid.json 2>/dev/null || true"),
        md("## 3. Fine-tune with the best setting (early stopping on validation loss)"),
        code("!python lugbara_mt.py train --model $MODEL"),
        md("## 4. Score on the test split"),
        code("!python lugbara_mt.py evaluate --model $MODEL\n!cat out/$MODEL/scores.json"),
        md("## 5. Plots: grid search, training curves, test scores, quality by sentence length"),
        code("!python lugbara_mt.py plots --model $MODEL\nP.show(sorted(Path(f'out/{MODEL}/figures').glob('*.png')))"),
        md("## 6. Model card and upload"),
        code("!python lugbara_mt.py card --model $MODEL --repo $REPO\n!python lugbara_mt.py push --model $MODEL --repo $REPO"),
    ]
    return cells


def local_model(name, nb):
    """The same model and settings as its SALT run, trained on the local corpus, then the 2 x 2 cross-test."""
    return [setup_md(nb["title"], nb["body"], nb["repo"], nb["hours"]), PIP, LOGIN,
            code(f"""
MODEL = "{nb["model"]}"
REPO = "{nb["repo"]}"
"""),
            *writefiles("lugbara_mt.py", "lugbara_plots.py"), *STRICT,
            md("## 1. Data: split the local corpus 70/15/15, prepare the SALT sample for the cross-tests, "
               "and analyse the Lugbara (with a comparison against SALT)"),
            code("""
!python lugbara_mt.py prepare --dataset local
!python lugbara_mt.py prepare
!python lugbara_mt.py analyze --dataset local
"""),
            md("### Local corpus analysis and comparison with SALT: charts"), SHOW_DATA,
            md("## 2. Fine-tune with the settings the SALT grid search chose (no new grid; early stopping on validation loss)"),
            code("!python lugbara_mt.py train --model $MODEL --no-grid"),
            md("## 3. Score on the local test split"),
            code("""
!python lugbara_mt.py evaluate --model $MODEL
!cat out/$MODEL/scores.json
"""),
            md("## 4. Cross-tests: this model on the SALT test split, the SALT-trained twin on the local test split, "
               "and the 2 × 2 table (trained on SALT / local × tested on SALT / local)"),
            code("""
!python lugbara_mt.py evaluate --model $MODEL --on salt --name on_salt
!python lugbara_mt.py evaluate --model $MODEL --counterpart --on local --name salt_model_on_local
!python lugbara_mt.py compare --model $MODEL
"""),
            md("## 5. Plots: training curves, test scores, quality by length, SALT vs local"),
            code("""
!python lugbara_mt.py plots --model $MODEL
P.show(sorted(Path(f'out/{MODEL}/figures').glob('*.png')))
"""),
            md("## 6. Model card and upload"),
            code("""
!python lugbara_mt.py card --model $MODEL --repo $REPO
!python lugbara_mt.py push --model $MODEL --repo $REPO
""")]


def fusion(name, nb):
    return [setup_md(nb["title"], nb["body"], nb["repo"], nb["hours"]), PIP, LOGIN,
            code(f'REPO = "{nb["repo"]}"'),
            *writefiles("lugbara_mt.py", "lugbara_plots.py", "fusion_mt.py"), *STRICT,
            *data_cells(),
            md("## 2. Train the fusion model (all four directions, early stopping on validation loss)"),
            code("!python fusion_mt.py train"),
            md("## 3. Score on the test split"),
            code("!python fusion_mt.py evaluate\n!cat out/fusion/scores.json"),
            md("## 4. Plots: training, test scores, quality by length, and fusion vs the single models"),
            code("!python fusion_mt.py plots\nP.show(sorted(Path('out/fusion/figures').glob('*.png')))"),
            md("## 5. Model card and upload"),
            code("!python fusion_mt.py card --repo $REPO\n!python fusion_mt.py push --repo $REPO")]


def fusion_local(name, nb):
    """The fusion network on the local corpus (English <-> Lugbara), then the 2 x 2 cross-test."""
    return [setup_md(nb["title"], nb["body"], nb["repo"], nb["hours"]), PIP, LOGIN,
            code(f'REPO = "{nb["repo"]}"'),
            *writefiles("lugbara_mt.py", "lugbara_plots.py", "fusion_mt.py"), *STRICT,
            md("## 1. Data: split the local corpus 70/15/15, prepare the SALT sample for the cross-tests, "
               "and analyse the Lugbara (with a comparison against SALT)"),
            code("""
!python lugbara_mt.py prepare --dataset local
!python lugbara_mt.py prepare
!python lugbara_mt.py analyze --dataset local
"""),
            md("### Local corpus analysis and comparison with SALT: charts"), SHOW_DATA,
            md("## 2. Train the fusion model (English ↔ Lugbara, same settings as on SALT, early stopping on validation loss)"),
            code("!python fusion_mt.py train --dataset local"),
            md("## 3. Score on the local test split"),
            code("""
!python fusion_mt.py evaluate --dataset local
!cat out/fusion_local/scores.json
"""),
            md("## 4. Cross-tests: this model on the SALT test split, the SALT-trained fusion model on the local "
               "test split, and the 2 × 2 table"),
            code("""
!python fusion_mt.py evaluate --dataset local --on salt --name on_salt
!python fusion_mt.py evaluate --dataset local --counterpart --on local --name salt_model_on_local
!python fusion_mt.py compare --dataset local
"""),
            md("## 5. Plots: training, test scores, quality by length, fusion vs the local single models, SALT vs local"),
            code("""
!python fusion_mt.py plots --dataset local
P.show(sorted(Path('out/fusion_local/figures').glob('*.png')))
"""),
            md("## 6. Model card and upload"),
            code("""
!python fusion_mt.py card --dataset local --repo $REPO
!python fusion_mt.py push --dataset local --repo $REPO
""")]


def report(name, nb):
    return [setup_md(nb["title"], nb["body"], None, nb["hours"], gpu=False),
            code('!pip install -q "transformers>=4.46,<5" sacrebleu matplotlib huggingface_hub'),
            *writefiles("lugbara_mt.py", "lugbara_plots.py"),
            md("## Download every published model's results and draw the charts"),
            code("""
import lugbara_plots as P, lugbara_mt as L
paths = P.report("report", thesis=L.THESIS)
print(len(paths), "figures")
"""),
            md("## Lugbara language analysis"),
            code("from pathlib import Path\nP.show(sorted(Path('report/figures').glob('data_*.png')))"),
            md("## Model comparison (same SALT test split)"),
            code("P.show(sorted(Path('report/figures').glob('compare_*.png')))"),
            md("## Per-model detail: grid search, training curves, test scores, quality by length"),
            code("P.show(sorted(p for p in Path('report/figures').glob('*.png') if not p.name.startswith(('data_', 'compare_'))))")]


NOTEBOOKS = {
    "01_xlmr_english_to_lugbara.ipynb": (single_model, {
        "model": "xlmr", "repo": "ckuma/lugbara-multilingual-translator", "copy_first": True,
        "title": "XLM-R: English → Lugbara and English → Luganda (v2)", "hours": "about 4–5 hours",
        "body": "The thesis model of *C. D. Akuma (2025), An Optimized Machine Translation Model for Low-Resource "
                "Lugbara Language* (Muni University), trained on 15,524 randomly drawn Sunbird SALT sentences "
                "(70/15/15 split, as in the thesis). Version 2: 2-pass grid probes, faster-learning cross-attention, "
                "label smoothing, patience 5, no repeat loops."}),
    "02_mt5_lugbara_to_english.ipynb": (single_model, {
        "model": "mt5", "repo": "ckuma/lugbara-english-translator", "copy_first": False,
        "title": "mT5: Lugbara → English and Luganda → English", "hours": "about 4–6 hours",
        "body": "The thesis model of *C. D. Akuma (2025)*, trained on 15,524 randomly drawn Sunbird SALT "
                "sentences (70/15/15 split, as in the thesis)."}),
    "03_fusion_xlmr_mt5.ipynb": (fusion, {
        "repo": "ckuma/lugbara-ensemble-translator", "title": "XLM-R + mT5 fusion: English ↔ Lugbara and English ↔ Luganda",
        "hours": "about 6–9 hours",
        "body": "An ensemble of the thesis's two models in one network: XLM-R and mT5 both encode the source, and "
                "mT5's decoder attends over both encoders' states to write the translation. One model, four "
                "directions, trained on the same SALT split so its scores compare directly with XLM-R and mT5."}),
    "04_results_report.ipynb": (report, {
        "title": "Lugbara MT: results report and charts", "hours": "a few minutes",
        "body": "Downloads the published results of XLM-R, mT5 and the fusion model from Hugging Face and draws "
                "the comparison charts. No GPU or token needed; models not yet published are skipped."}),
    "05_mt5_local_corpus.ipynb": (local_model, {
        "model": "mt5_local", "repo": "ckuma/lugbara-english-translator-local",
        "title": "mT5 on the local corpus: Lugbara → English, compared with SALT", "hours": "about 2–3 hours",
        "body": "The thesis's Lugbara→English model (mT5) trained on the local English–Lugbara corpus published "
                "with the thesis ([Akuma1992/NLP-data-set](https://github.com/Akuma1992/NLP-data-set)), with the "
                "settings its SALT run chose (learning rate 1e-3, batch 16; no new grid). It is then tested on both "
                "test splits, and the SALT-trained mT5 ([ckuma/lugbara-english-translator]"
                "(https://huggingface.co/ckuma/lugbara-english-translator)) is tested on the local split, so the "
                "two corpora can be compared like for like."}),
    "06_xlmr_local_corpus.ipynb": (local_model, {
        "model": "xlmr_local", "repo": "ckuma/lugbara-multilingual-translator-local",
        "title": "XLM-R on the local corpus: English → Lugbara, compared with SALT", "hours": "about 2–3 hours",
        "body": "The thesis's English→Lugbara model (XLM-R encoder-decoder) trained on the local English–Lugbara "
                "corpus published with the thesis ([Akuma1992/NLP-data-set](https://github.com/Akuma1992/NLP-data-set)), "
                "with the settings of the first SALT-trained XLM-R (learning rate 3e-5, batch 16; no new grid). It is "
                "then tested on both test splits, and that SALT-trained XLM-R (revision b2eec05b of "
                "[ckuma/lugbara-multilingual-translator](https://huggingface.co/ckuma/lugbara-multilingual-translator)) "
                "is tested on the local split."}),
    "07_fusion_local_corpus.ipynb": (fusion_local, {
        "repo": "ckuma/lugbara-ensemble-translator-local",
        "title": "XLM-R + mT5 fusion on the local corpus: English ↔ Lugbara, compared with SALT",
        "hours": "about 4–6 hours",
        "body": "The fusion of the thesis's two models (XLM-R and mT5 both encode the source; mT5's decoder attends "
                "over both) trained on the local English–Lugbara corpus published with the thesis "
                "([Akuma1992/NLP-data-set](https://github.com/Akuma1992/NLP-data-set)) in both directions, with the "
                "settings of the SALT-trained fusion model. It is then tested on both test splits, and the SALT-trained "
                "fusion model ([ckuma/lugbara-ensemble-translator](https://huggingface.co/ckuma/lugbara-ensemble-translator)) "
                "is tested on the local split."}),
}

for name, (fn, nb) in NOTEBOOKS.items():
    gpu = fn is not report
    out = {"cells": with_ids(fn(name, nb)), "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True, "isGpuEnabled": gpu}},
        "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", HERE / name, "-", len(out["cells"]), "cells")
