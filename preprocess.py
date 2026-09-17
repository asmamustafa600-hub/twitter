"""
preprocess.py
-------------
Loads the Twitter sentiment CSVs and cleans the tweet text.
Shared by both the Naive Bayes and RNN training scripts, and by the app.
"""

import re
import pandas as pd

COLUMN_NAMES = ["id", "entity", "sentiment", "text"]


def load_raw(train_path, val_path):
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    val_df = pd.read_csv(val_path, header=None, names=COLUMN_NAMES)
    return train_df, val_df


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/punctuation/extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"@\w+", " ", text)                  # mentions
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    text = re.sub(r"\s+", " ", text).strip()            # extra whitespace
    return text


def prepare_dataset(train_path, val_path):
    """Returns cleaned, deduplicated, non-empty train/val DataFrames."""
    train_df, val_df = load_raw(train_path, val_path)

    for df in (train_df, val_df):
        df.dropna(subset=["text"], inplace=True)
        df["clean_text"] = df["text"].apply(clean_text)
        df.drop_duplicates(subset=["clean_text"], inplace=True)
        df.query("clean_text != ''", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df
