# Startup Guide

How to set up the project and what each script does. Run the scripts in this order
the first time — each one depends on artifacts produced by the previous step.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 1. `download_data.py`

Downloads the [Sarcasm News Headline Dataset](https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline)
from HuggingFace and caches it as `data/train.csv` and `data/test.csv`. Skips the
download if those files already exist.

```bash
python download_data.py
```

## 2. `data_processing.py`

Not run directly as part of the pipeline — it's a library imported by `train.py`,
`evaluate.py`, and `tune.py`. It exposes:

- `clean_text(text)` — lowercases, strips punctuation, removes stopwords
- `load_data(max_features=10_000)` — loads the CSVs, cleans them, fits a TF-IDF
  vectorizer on the training set, and returns `X_train, X_test, y_train, y_test, vectorizer`

You can run it standalone as a sanity check — it prints matrix shapes and class
balance without training anything:

```bash
python data_processing.py
```

## 3. `train.py`

Trains a `LogisticRegression` (SAGA solver, L1 penalty) on TF-IDF features and saves
two files: `model.joblib` and `vectorizer.joblib`. Both are required by every script
that follows.

```bash
python train.py
```

## 4. `evaluate.py`

Loads `model.joblib` and `vectorizer.joblib`, runs the model against the held-out
test set, prints a precision/recall/F1 classification report, and saves
`confusion_matrix.png`.

```bash
python evaluate.py
```

## 5. `tune.py`

Grid-searches the TF-IDF `max_features` cap (retraining a fresh model for each
value), prints F1 per value, and saves `f1_vs_max_features.png`. Useful for
re-justifying the `max_features` default in `train.py` if the dataset or cleaning
logic changes.

```bash
python tune.py
```

## 6. `api.py`

Loads the saved model and vectorizer once at startup and serves a FastAPI prediction
endpoint at `http://127.0.0.1:8000`. Requires `model.joblib` and `vectorizer.joblib`
to already exist (run `train.py` first).

```bash
python api.py
```

Interactive docs: open `http://127.0.0.1:8000/docs` in a browser and try `POST /predict`
directly.

Or via curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"headline\": \"Local Man Discovers Sarcasm, Nation Stunned\"}"
```

## 7. `test_cases.py`

Sends a curated set of headlines (obvious sarcasm, obvious factual news, subtle
sarcasm, sarcasm-adjacent-but-neutral, short headlines, headlines with numbers/proper
nouns) to the running API and prints expected vs. predicted labels with confidence.
Requires `api.py` to already be running in another terminal.

```bash
python test_cases.py
```

## Full Sequence

```bash
python download_data.py
python train.py
python evaluate.py
python tune.py
python api.py          # leave running in this terminal
python test_cases.py   # run in a second terminal
```
