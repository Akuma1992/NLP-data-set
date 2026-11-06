# Lugbara translator

English ↔ Lugbara (and Luganda) machine translation following **C. D. Akuma, *An Optimized
Machine Translation Model for Low-Resource Lugbara Language*, MSc Computer Science, Muni
University (October 2025)**: XLM-R for English → Lugbara, mT5 for Lugbara → English, and an
XLM-R + mT5 fusion of the two. Training runs on Kaggle GPUs; the trained models are published
on Hugging Face under [ckuma](https://huggingface.co/ckuma).

## Trained models

Model weights are on Hugging Face (they are too large for GitHub):

| Model | Directions | Trained on | Hugging Face | Test BLEU |
|---|---|---|---|---|
| mT5 | Lugbara → English, Luganda → English | SALT sample | [ckuma/lugbara-english-translator](https://huggingface.co/ckuma/lugbara-english-translator) | 12.71 / 16.65 |
| XLM-R + mT5 fusion | English ↔ Lugbara, English ↔ Luganda | SALT sample | [ckuma/lugbara-ensemble-translator](https://huggingface.co/ckuma/lugbara-ensemble-translator) | en→lgg 8.14, en→lug 6.72, lgg→en 5.72, lug→en 6.57 |
| XLM-R | English → Lugbara, English → Luganda | SALT sample | [ckuma/lugbara-multilingual-translator](https://huggingface.co/ckuma/lugbara-multilingual-translator) | v1 (revision `b2eec05b`) 0.29 / 0.16; the current v2 did not converge |
| mT5 | Lugbara → English | local corpus (this repository) | `ckuma/lugbara-english-translator-local` | training |
| XLM-R | English → Lugbara | local corpus (this repository) | `ckuma/lugbara-multilingual-translator-local` | training |
| XLM-R + mT5 fusion | English ↔ Lugbara | local corpus (this repository) | `ckuma/lugbara-ensemble-translator-local` | to do |

For English → Lugbara the fusion model is the one to use. Each model card has the full scores
(BLEU, TER, chrF, word F1, apostrophe integrity), charts, sample translations and the Lugbara
language analysis of its training data.

## Data

* **SALT** — [Sunbird/salt](https://huggingface.co/datasets/Sunbird/salt) `text-all` (CC BY-SA 4.0;
  the Hugging Face account must accept its terms). The thesis corpus size is reproduced with a
  random sample of 15,524 sentences, split 70/15/15 (10,867 / 2,329 / 2,328), seed 13, lowercased.
* **Local corpus** — [`datasetnlp .csv`](../datasetnlp%20.csv) in this repository (15,525 rows).
  After removing 1,163 empty, 60 duplicate and 25 malformed rows: 14,277 English–Lugbara pairs,
  split 70/15/15 at random (9,994 / 2,142 / 2,141), seed 13, with the same cleaning as SALT.
* 6,109 of the local pairs have an English sentence that is also in the SALT sample, so the
  SALT-vs-local comparison is reported both on all test sentences and on the sentences the two
  corpora do not share.

## Layout

| Path | What it is |
|---|---|
| `kaggle/lugbara_mt.py` | The pipeline: `prepare`, `grid`, `train`, `evaluate`, `compare`, `analyze`, `plots`, `card`, `fetch`, `push`, `smoke` |
| `kaggle/fusion_mt.py` | The XLM-R + mT5 fusion model (`--dataset salt` or `--dataset local`) |
| `kaggle/lugbara_plots.py` | Charts for the notebooks and model cards |
| `kaggle/build_notebooks.py` | Builds the Kaggle notebooks below from the three scripts |
| `kaggle/01`–`04` | XLM-R, mT5 and fusion on SALT; results report |
| `kaggle/05`–`07` | mT5, XLM-R and fusion on the local corpus, each cross-tested against its SALT-trained twin |
| `kaggle/08` | After 05 and 06: the comparison on the sentences the two corpora do not share |
| `results/` | What the SALT runs produced: scores, training logs, grid searches, charts, test translations and the published model cards |
| `scripts/` | The first approach (extending the MarianMT Rutooro translator; `train.py`, `prepare_data.py`, `evaluate.py`, `common.py`), the Lugbara analysis, and helpers (`rebuild_data.py` rebuilds the SALT sample on a PC) |

## Running

On Kaggle: import a notebook from `kaggle/`, choose **GPU T4 x2**, turn **Internet** on, attach a
secret **`HF_TOKEN`** (a Hugging Face *write* token of an account that accepted the SALT terms),
then **Save Version → Save & Run All (Commit)**. Each notebook uploads its model and card when it
finishes.

A quick CPU check of the code (small base models in `base-models/`; set `LUGBARA_LOCAL_CSV` to read
the CSV from disk instead of GitHub):

```bash
python kaggle/lugbara_mt.py smoke --model mt5_local --base-root base-models
```

## Credits

* Thesis, method and the local corpus: Charity Driciru Akuma, Muni University (2025).
* SALT: Sunbird AI (CC BY-SA 4.0).
* Base models: `google/mt5-base`, `FacebookAI/xlm-roberta-base`.
* Earlier work: [mmuniversity/rutooro-multilingual-translator](https://huggingface.co/mmuniversity/rutooro-multilingual-translator) (Mountains of the Moon University) and Helsinki-NLP.
* Code: Samuel (samocenuel).
