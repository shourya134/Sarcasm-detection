import os
from datasets import load_dataset


DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


def download():
    """
    Download the Sarcasm News Headline dataset from HuggingFace and save
    train/test splits as CSV files under the data/ directory.

    Skips the download if files already exist.

    Example:
        $ python download_data.py
        Downloading dataset from HuggingFace...
        Saved 28619 train rows to data/train.csv
        Saved 26709 test rows to data/test.csv
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

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved {len(train_df)} train rows to {TRAIN_PATH}")
    print(f"Saved {len(test_df)} test rows to {TEST_PATH}")


if __name__ == "__main__":
    download()
