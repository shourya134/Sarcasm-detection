import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import clean_text, load_data, DATA_DIR

DATA_AVAILABLE = os.path.exists(os.path.join(DATA_DIR, "train.csv")) and os.path.exists(
    os.path.join(DATA_DIR, "test.csv")
)


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_strips_punctuation():
    assert clean_text("Wow, really?!") == "wow really"


def test_clean_text_removes_stopwords():
    # "is", "the", "a" are common English stopwords and should be dropped
    assert clean_text("This is the best") == "best"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


@pytest.mark.skipif(not DATA_AVAILABLE, reason="data/train.csv and data/test.csv not found — run download_data.py first")
def test_load_data_returns_consistent_feature_dimensions():
    X_train, X_test, y_train, y_test, vectorizer = load_data(max_features=1_000)

    # train and test must share the same vocabulary/feature space, or the model
    # trained on X_train cannot be scored against X_test
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] <= 1_000

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="data/train.csv and data/test.csv not found — run download_data.py first")
def test_load_data_labels_are_binary():
    _, _, y_train, _, _ = load_data(max_features=1_000)
    assert set(y_train.unique()).issubset({0, 1})
