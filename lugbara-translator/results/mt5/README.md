---
language: [en, lg, lgg]
license: apache-2.0
library_name: transformers
pipeline_tag: translation
base_model: google/mt5-base
datasets: [Sunbird/salt]
metrics: [bleu, ter, chrf]
tags: [translation, lugbara, luganda, low-resource, uganda, muni-university]
---

# mT5 Lugbara/Luganda→English translator

Fine-tuned `google/mt5-base` for Lugbara → English, Luganda → English.
It follows the model choice and training method of **C. D. Akuma, *An Optimized Machine
Translation Model for Low-Resource Lugbara Language*, MSc Computer Science, Muni University
(October 2025)**: mT5 was the thesis's optimised Lugbara→English model.

## How to use

Put the target-language tag in front of the sentence and use lowercase text, as in training.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("ckuma/lugbara-english-translator")
model = AutoModelForSeq2SeqLM.from_pretrained("ckuma/lugbara-english-translator")

def translate(text, tag):
    x = tok(tag + " " + text.lower(), return_tensors="pt")
    y = model.generate(**x, num_beams=4, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)

print(translate("ba erini tutuni o'duko oku eyini azini ezoanzi eyini ri e'yo peta pezu 'diyi ma alea ku", "<2en>"))
print(translate("ba ma 'ye mutukari ma rua e'deta eri 'bazu pa dria 'diyi ra ka de nga fu ni mu amve azia ku", "<2en>"))
```

Tags: `<2en>` for Lugbara → English, `<2en>` for Luganda → English.

## Results (test split, 4656 sentence pairs)

| Direction | Sentences | BLEU ↑ | TER ↓ | Word F1 ↑ | chrF ↑ | Apostrophe integrity ↑ | Thesis best BLEU* |
|---|---|---|---|---|---|---|---|
| Lugbara → English | 2328 | 12.71 | 73.47 | 0.3942 | 34.73 | – | 25.41 (mT5 (tuned)) |
| Luganda → English | 2328 | 16.65 | 67.81 | 0.4455 | 38.88 | – | – |

BLEU and TER are sacreBLEU corpus scores; word F1 is unigram precision/recall F1; apostrophe
integrity is the share of apostrophe words in the reference (`'ba`, `a'dule` …) reproduced intact.
\*The thesis scores were measured on the thesis's own corpus, not on SALT, so they are a
reference point rather than a like-for-like comparison.

### Sample test translations (Lugbara → English)

| Source | Reference | Model |
|---|---|---|
| ba erini tutuni o'duko oku eyini azini ezoanzi eyini ri e'yo peta pezu 'diyi ma alea ku | women and girls' voices are often excluded from decision making. | usually, women and girls are not involved in decision-making. |
| ba ma 'ye mutukari ma rua e'deta eri 'bazu pa dria 'diyi ra ka de nga fu ni mu amve azia ku | the cars have to be serviced before they go out to the field. | car accidents should be avoided. |
| ba lu azi'ba gameteni kari uganda niri ma alea 'diyi ma azi eyini azi fefe | some of the government officials in uganda are just appointed into office. | some of the government officials in uganda have jobs. |
| bongo azi'ba yi ageyi tepiri yini suu ri nzi nzi dika dri dri. | the personal protective equipment is heavy and hot. | clothes and gloves are ready to be opened. |
| drice'ba azo korona niri ni 'diyi o'bi azo eyi ma ereta distrikitia ri atri ra | the task force managed to contain the spread of diseases in the district. | the ministry prevents the spread of diseases in the district. |
| pari ndea 'ba leni 'ba ma ca ku | the area is out of bounds. | the area is not a ghost. |
| e'yo ecopi 'ba ode eyi ma zota ewu drialeri ma alea ri atri pi ra 'diyi a'duni? | what could be the hindrances to the future development of youths? | what are the causes of youths' growth? |
| ma a'bi dra de eri ma eli kali aro drini aro. | my grandfather died when he was eighty eight years old. | my grandfather died when he was eight years old. |

## Training

* **Data:** the thesis corpus (15,524 English–Lugbara pairs) is not public, so the same number
  of sentences was drawn at random (seed 13) from the cleaned [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt)
  `text-all` table and split 70/15/15 like the thesis: 10,867 train,
  2,329 validation, 2,328 test. Luganda uses
  the same sentences. SALT has no domain labels, so the split is random rather than stratified by domain.
* **Preprocessing (thesis §3.2.2):** lowercased, punctuation normalised, all apostrophe variants
  folded to `'`, empty / duplicate / malformed pairs removed.
* **Model:** mT5 base, sequence-to-sequence.
* **Hyperparameters (thesis §3.4.3):** grid search over learning rate × batch size
  (0.0003/8 → 3.79 BLEU, 0.0003/16 → 1.4 BLEU, 0.0005/8 → 3.05 BLEU, 0.0005/16 → 1.68 BLEU, 0.001/8 → 0.27 BLEU, 0.001/16 → 4.11 BLEU),
  then fine-tuning with lr 0.001, batch 16, optimiser adafactor,
  early stopping on validation loss (patience 3); 8,154 steps,
  83.5 minutes on a Kaggle GPU.

See [LUGBARA_ANALYSIS.md](LUGBARA_ANALYSIS.md) for the Lugbara language analysis of the training data.

## Limits

* Trained on short SALT sentences (agriculture, health, community life); other domains are weaker.
* Lowercase only. Tone is not written in Lugbara, so tone-only differences are lost.
* Automatic scores use one reference per sentence; Lugbara speakers should review output before use.

## Credits

* Thesis and method: Charity Driciru Akuma, Muni University (2025).
* Data: Sunbird AI, SALT (CC BY-SA 4.0).
* Base model: `google/mt5-base`.
* Earlier work: [mmuniversity/rutooro-multilingual-translator](https://huggingface.co/mmuniversity/rutooro-multilingual-translator) (Mountains of the Moon University).
