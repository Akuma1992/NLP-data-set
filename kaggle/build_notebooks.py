"""Build the two Kaggle notebooks from lugbara_mt.py (run again after editing the script)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = (HERE / "lugbara_mt.py").read_text(encoding="utf-8")

NOTEBOOKS = {
    "01_xlmr_english_to_lugbara.ipynb": {
        "model": "xlmr", "repo": "ckuma/lugbara-multilingual-translator", "copy_first": True,
        "title": "XLM-R: English → Lugbara and English → Luganda",
        "hours": "about 2–3 hours",
    },
    "02_mt5_lugbara_to_english.ipynb": {
        "model": "mt5", "repo": "ckuma/lugbara-english-translator", "copy_first": False,
        "title": "mT5: Lugbara → English and Luganda → English",
        "hours": "about 4–6 hours",
    },
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build(name, nb):
    cells = [
        md(f"""
# {nb['title']}

The thesis model of *C. D. Akuma (2025), An Optimized Machine Translation Model for Low-Resource
Lugbara Language* (Muni University), trained on 15,524 randomly drawn Sunbird SALT sentences
(70/15/15 split, as in the thesis) and published to **[{nb['repo']}](https://huggingface.co/{nb['repo']})**.

**Before you run it** (right-hand panel → *Settings* and *Add-ons*):
1. *Accelerator*: **GPU T4 x2** (only one GPU is used, so the batch sizes match the thesis).
2. *Internet*: **on** (needs a phone-verified Kaggle account).
3. *Add-ons → Secrets*: add a secret named **`HF_TOKEN`** holding the Hugging Face **Write** token
   of the account that owns `ckuma` and has accepted the SALT terms, and tick it for this notebook.

Then **Save Version → Save & Run All (Commit)**. It runs in the background ({nb['hours']}) and
uploads the model when it finishes; the browser can be closed.
"""),
        code('!pip install -q "transformers>=4.46,<5" "datasets>=2.19" sacrebleu sentencepiece protobuf accelerate'),
        code("""
import re
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami
raw = UserSecretsClient().get_secret("HF_TOKEN") or ""
found = re.findall(r"hf_[A-Za-z0-9]{30,}", raw)
print("HF_TOKEN secret check (the token itself is never printed):", len(raw), "characters; starts with hf_:", raw.strip().startswith("hf_"), "; contains '...':", "..." in raw, "; full-length tokens found:", len(found))
login(token=found[0] if found else raw.strip(), add_to_git_credential=False)
print("Signed in to Hugging Face as", whoami()["name"])
"""),
        code(f"""
MODEL = "{nb['model']}"
REPO = "{nb['repo']}"
RUN_GRID = True      # thesis 3.4.3: learning-rate x batch-size grid search before the final run
"""),
        md("The full pipeline script (data sampling, grid search, training, scoring, analysis, card, upload)."),
        code("%%writefile lugbara_mt.py\n" + SCRIPT),
        md("Stop the whole run as soon as any step fails (a failing `!command` does not stop a notebook by itself)."),
        code("""
ip = get_ipython()
_system = ip.system
def _strict_system(cmd):
    _system(cmd)
    if ip.user_ns.get("_exit_code", 0):
        raise RuntimeError("step failed: " + cmd)
ip.system = _strict_system
print("steps will stop the run on failure")
"""),
    ]
    if nb["copy_first"]:
        cells += [md("Copy the Rutooro model into the repo first; the trained model replaces it at the end."),
                  code("!python lugbara_mt.py copy --repo $REPO")]
    cells += [
        md("## 1. Data: sample and split SALT like the thesis"),
        code("!python lugbara_mt.py prepare\n!python lugbara_mt.py analyze"),
        md("## 2. Grid search (learning rate × batch size)"),
        code("if RUN_GRID:\n    !python lugbara_mt.py grid --model $MODEL\n!cat out/$MODEL/grid.json 2>/dev/null || true"),
        md("## 3. Fine-tune with the best setting (early stopping on validation loss)"),
        code("!python lugbara_mt.py train --model $MODEL"),
        md("## 4. Score on the test split"),
        code("!python lugbara_mt.py evaluate --model $MODEL\n!cat out/$MODEL/scores.json"),
        md("## 5. Model card and upload"),
        code("!python lugbara_mt.py card --model $MODEL --repo $REPO\n!python lugbara_mt.py push --model $MODEL --repo $REPO"),
    ]
    out = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", HERE / name)


for name, nb in NOTEBOOKS.items():
    build(name, nb)
