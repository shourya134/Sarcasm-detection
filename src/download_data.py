import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def download():
    """
    Download the Sarcasm News Headline dataset from HuggingFace and save
    train/test splits as CSV files under the data/ directory.

    The dataset's own train/test split has heavy overlap (~99.6% of "test" headlines
    also appear in "train"), which leaks test data into training and inflates every
    downstream metric. To avoid this, both splits are combined, deduplicated by
    headline, and re-split into a fresh stratified train/test set.

    Skips the download if files already exist.

    Example:
        $ python download_data.py
        Downloading dataset from HuggingFace...
        Saved 22140 train rows to data/train.csv
        Saved 5535 test rows to data/test.csv
    """
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        print("Data already downloaded, skipping.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # HuggingFace dataset: https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline
    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset("raquiba/Sarcasm_News_Headline")

    # Drop article_link — not useful for text classification
    train_df = dataset["train"].to_pandas()[["headline", "is_sarcastic"]]
    test_df = dataset["test"].to_pandas()[["headline", "is_sarcastic"]]

    # Combine and deduplicate before splitting, since the dataset's own split
    # overlaps heavily and would otherwise leak test headlines into training
    all_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(subset="headline")

    train_df, test_df = train_test_split(
        all_df, test_size=TEST_SIZE, stratify=all_df["is_sarcastic"], random_state=RANDOM_STATE
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved {len(train_df)} train rows to {TRAIN_PATH}")
    print(f"Saved {len(test_df)} test rows to {TEST_PATH}")


if __name__ == "__main__":
    download()
