# Multilabel Bengali Toxicity Classification

BanglaBERT + BanglaCAT intermediate transfer learning + TF-IDF ensemble for the
**Bengali NLP Safety Challenge 2026** (7 independent binary labels per comment:
`is_toxic`, `implicit_toxicity`, `sarcasm`, `contains_profanity`, `threatening`,
`identity_attack`, `sexual_content`).

Everything lives in one notebook: [`bengali_toxicity_multilabel.ipynb`](bengali_toxicity_multilabel.ipynb).

## Approach

1. Conservative whitespace normalisation (script, punctuation and spelling variants are kept).
2. **Baseline:** word + character TF-IDF, one logistic regression per label.
3. **Transfer learning:** fine-tune `csebuetnlp/banglabert` on BanglaCAT's 12-way label
   (train split only), keep the **encoder only**, discard the 12-way head.
4. Fine-tune that encoder on the competition data with a new 7-output head and
   class-weighted `BCEWithLogitsLoss`.
5. Choose the checkpoint, the BanglaBERT/TF-IDF blend weight and one threshold per label on the validation split.
6. Retrain on train + validation and predict the test set.

## Validation results

| Model | Macro-F1 | Micro-F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.4815 | 0.6272 |
| BanglaBERT (BanglaCAT-adapted encoder) | 0.5488 | 0.6485 |
| **Ensemble (0.45 BanglaBERT + 0.55 TF-IDF)** | **0.5722** | 0.6539 |

![Training curves](figures/training_curves.png)
![Per-label F1](figures/per_label_f1.png)

Caveats: the blend weight and thresholds were tuned on the same 500 validation examples, so these
scores are optimistic; rare labels have only 15–24 validation positives; and the notebook has no
BanglaCAT on/off comparison, so the benefit of the transfer stage is not measured here
(set `USE_INTERMEDIATE_PRETRAINING = False` to run it).

## Data (not included)

```text
data/
├── train.csv  validation.csv  test.csv  sample_submission.csv   # competition files
└── banglacat/bangla_context_toxicity_train.csv                  # BanglaCAT train split
```

- **Competition data:** join *Bengali NLP Safety Challenge 2026* on Kaggle, accept the rules, then download from the
  Data tab or run `kaggle competitions download -c bengali-nlp-safety-challenge-2026`.
- **BanglaCAT:** public dataset on Mendeley Data (search "BanglaCAT"); accept its license, cite it as its page
  requests, and copy only the **train** CSV into `data/banglacat/`.
- Use a different location with the `DATA_DIR` environment variable.

The datasets contain toxic, abusive and sexual language. Do not commit them or print raw records.

## Running

```bash
pip install torch transformers accelerate sentencepiece scikit-learn scipy pandas matplotlib tqdm jupyter
jupyter notebook bengali_toxicity_multilabel.ipynb
```

Reported results used one Tesla T4 GPU. Outputs (adapted encoder, `submission.csv`, `model_config.json`)
are written to `outputs/`; figures to `figures/`.
