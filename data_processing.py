import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# quiet=True prevents spamming output on every run if stopwords are already downloaded
nltk.download("stopwords", quiet=True)

DATA_DIR = "data"
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Lowercase, remove punctuation, and strip stopwords from a headline.

    Example:
        >>> clean_text("Oh wow, that's TOTALLY not sarcastic at all!")
        'wow thats totally sarcastic'
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def load_data():
    """
    Load train/test CSVs, clean headlines, and return TF-IDF feature matrices.

    Fits the TF-IDF vectorizer on training data only to prevent data leakage.
    Uses unigrams and bigrams with a vocabulary capped at 10,000 features.

    Returns:
        X_train, X_test (scipy sparse matrices), y_train, y_test (pd.Series)

    Example:
        >>> X_train, X_test, y_train, y_test = load_data()
        >>> X_train.shape
        (28619, 10000)
    """
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    # map() is faster than apply() for element-wise string operations
    train_df["headline"] = train_df["headline"].map(clean_text)
    test_df["headline"] = test_df["headline"].map(clean_text)

    # Fit on train only — transforming test with train vocabulary prevents data leakage
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
    X_train = vectorizer.fit_transform(train_df["headline"])
    X_test = vectorizer.transform(test_df["headline"])

    y_train = train_df["is_sarcastic"]
    y_test = test_df["is_sarcastic"]

    return X_train, X_test, y_train, y_test


# Sanity check — run directly to verify shapes and class balance before training
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Label distribution (train):\n{y_train.value_counts()}")
