# Results of the SALT runs

What the Kaggle notebooks produced for the three models trained on the Sunbird SALT sample
(15,524 sentences, 70/15/15 split, seed 13). The model weights themselves are on Hugging Face.

| Folder | Model | Hugging Face | Test BLEU |
|---|---|---|---|
| [`mt5/`](mt5/) | mT5, Lugbara → English and Luganda → English | [ckuma/lugbara-english-translator](https://huggingface.co/ckuma/lugbara-english-translator) | 12.71 / 16.65 |
| [`fusion/`](fusion/) | XLM-R + mT5 fusion, all four directions | [ckuma/lugbara-ensemble-translator](https://huggingface.co/ckuma/lugbara-ensemble-translator) | en→lgg 8.14, en→lug 6.72, lgg→en 5.72, lug→en 6.57 |
| [`xlmr/`](xlmr/) | XLM-R version 2, English → Lugbara and English → Luganda (did not converge) | [ckuma/lugbara-multilingual-translator](https://huggingface.co/ckuma/lugbara-multilingual-translator) | 0.0 / 0.0 (version 1, revision `b2eec05b`: 0.29 / 0.16) |

Each folder holds:

* `README.md`: the model card as published on Hugging Face
* `scores.json`: BLEU, TER, chrF, word F1 and apostrophe integrity on the 2,328-sentence test split
* `train_log.json`: learning rate, batch size, steps, minutes and the training/validation loss history
* `grid.json`: the learning-rate × batch-size grid search (mT5, XLM-R)
* `LUGBARA_ANALYSIS.md`, `lugbara_analysis.json`: the Lugbara language analysis of the training data
* `figures/`: the charts shown in the notebooks and model cards
* `test_outputs/`: every test sentence with its reference and the model's translation (fusion, XLM-R)

The test sentences and references come from [Sunbird SALT](https://huggingface.co/datasets/Sunbird/salt)
(Sunbird AI, CC BY-SA 4.0); reuse them under that licence with attribution.

Results of the models trained on this repository's `datasetnlp .csv` will be on their model cards
(`ckuma/lugbara-english-translator-local`, `ckuma/lugbara-multilingual-translator-local`,
`ckuma/lugbara-ensemble-translator-local`) when the Kaggle runs finish.
