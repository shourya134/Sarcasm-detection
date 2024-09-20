# Startup Guide

How to set up the project and what each script does. Run the scripts in this order
the first time — each one depends on artifacts produced by the previous step.

All scripts live under `src/` and are run as modules from the repo root using
`python -m src.<script_name>`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 1. `src/download_data.py`

Downloads the [Sarcasm News Headline Dataset](https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline)
from HuggingFace and caches it as `data/train.csv` and `data/test.csv`. Skips the
download if those files already exist.

```bash
python -m src.download_data
```

## 2. `src/data_processing.py`

Not run directly as part of the pipeline — it's a library imported by `src/train.py`,
`src/evaluate.py`, and `src/tune.py`. It exposes:

- `clean_text(text)` — lowercases, strips punctuation, removes stopwords
- `load_data(max_features=10_000)` — loads the CSVs, cleans them, fits a TF-IDF
  vectorizer on the training set, and returns `X_train, X_test, y_train, y_test, vectorizer`

You can run it standalone as a sanity check — it prints matrix shapes and class
balance without training anything:

```bash
python -m src.data_processing
```

## 3. `src/train.py`

Trains a `LogisticRegression` (SAGA solver, L1 penalty) on TF-IDF features and saves
two files: `artifacts/model.joblib` and `artifacts/vectorizer.joblib`. Both are
required by every script that follows.

```bash
python -m src.train
```

## 4. `src/evaluate.py`

Loads `artifacts/model.joblib` and `artifacts/vectorizer.joblib`, runs the model
against the held-out test set, prints a precision/recall/F1 classification report,
and saves `eval_results/confusion_matrix.png`.

```bash
python -m src.evaluate
```

## 5. `src/tune.py`

Grid-searches the TF-IDF `max_features` cap (retraining a fresh model for each
value), prints F1 per value, and saves `tuning_results/f1_vs_max_features.png`.
Useful for re-justifying the `max_features` default in `src/train.py` if the dataset
or cleaning logic changes.

```bash
python -m src.tune
```

## 6. `src/api.py`

Loads the saved model and vectorizer once at startup and serves a FastAPI prediction
endpoint at `http://127.0.0.1:8000`. Requires `artifacts/model.joblib` and
`artifacts/vectorizer.joblib` to already exist (run `src/train.py` first).

```bash
python -m src.api
```

Interactive docs: open `http://127.0.0.1:8000/docs` in a browser and try `POST /predict`
directly.

Or via curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"headline\": \"Local Man Discovers Sarcasm, Nation Stunned\"}"
```

## 7. `src/test_cases.py`

Sends a curated set of headlines (obvious sarcasm, obvious factual news, subtle
sarcasm, sarcasm-adjacent-but-neutral, short headlines, headlines with numbers/proper
nouns) to the running API and prints expected vs. predicted labels with confidence.
Requires `src/api.py` to already be running in another terminal.

```bash
python -m src.test_cases
```

## Tests

`tests/` contains unit tests for `clean_text()` and `load_data()`
(`tests/test_data_processing.py`) and the `/predict` endpoint
(`tests/test_api.py`). Tests that depend on data or trained artifacts skip
automatically (rather than failing) if `data/` or `artifacts/` aren't populated yet.

```bash
pytest tests/ -v
```

## Full Sequence

```bash
python -m src.download_data
python -m src.train
python -m src.evaluate
python -m src.tune
python -m src.api          # leave running in this terminal
python -m src.test_cases   # run in a second terminal
```
