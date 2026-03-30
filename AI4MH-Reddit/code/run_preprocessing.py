import os
import re
import argparse
import pandas as pd
from preprocessing import TextPreprocessor

# Constants

LABEL_COLS = ['functional']

LABEL_MAP = {
    1:   "positive",
    0:   "neutral",
   -1:   "negative",
    99:  "not_mentioned"
}

URL_ONLY_PATTERN = re.compile(r"^\s*(https?://\S+|www\.\S+)\s*$", re.IGNORECASE)
BATCH_SIZE = 1000

# Helper Functions

def is_image_post(text) -> bool:
    if not isinstance(text, str):
        return False
    return bool(URL_ONLY_PATTERN.match(text.strip()))


def is_removed(text) -> bool:
    if not isinstance(text, str):
        return True
    return text.strip().lower() in {"[removed]", "[deleted]", "[deleted by user]", ""}


def too_short_text(text) -> bool:
    return len(str(text).split()) < 5


def post_qualitycheck(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df["is_removed"] = df[text_col].apply(is_removed)
    df["is_image"]   = df[text_col].apply(is_image_post)
    print(f"  Removed/deleted : {df['is_removed'].sum()}")
    print(f"  Image-only      : {df['is_image'].sum()}")
    return df


def run_in_batches(p: TextPreprocessor, texts: list, batch_size: int = BATCH_SIZE) -> list:
    results = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        results.extend(p.run_fullcorpus(batch))
        print(f"  Processed {end} / {total} rows...")

    print(f"Text preprocessing complete for {total} rows.")
    return results


# Main Function

def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing once for AI mental health Reddit study."
    )
    parser.add_argument("--train_path", 
                        type=str,
                        default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/train.csv")
    parser.add_argument("--test_path", 
                        type=str,
                        default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/test.csv")
    args, _ = parser.parse_known_args()

    p = TextPreprocessor(
        use_lemma           = True,
        handle_negation     = True,
        expand_contractions = True,
        handle_emojis       = True,
        handle_chatwords    = True,
        remove_stopwords    = True,
        spell_check         = False,
    )
    print(f"Preprocessor: {p}\n")

    # Training data 
    print(f"Loading training data from {args.train_path}...")
    train_df = pd.read_csv(args.train_path)
    print(f"  Shape: {train_df.shape}")

    assert "text" in train_df.columns, "Expected a 'text' column in training data."
    for col in LABEL_COLS:
        assert col in train_df.columns, f"Expected label column '{col}' in training data."

    train_df = post_qualitycheck(train_df, "text")

    print("  Running TextPreprocessor on training data...")
    # Train is small (~800 rows) — no batching needed
    train_df['cleaned_text'] = p.run_fullcorpus(train_df["text"].tolist())
    train_df["too_short"]    = train_df['cleaned_text'].apply(too_short_text)

    print(f"  Flagged {train_df['too_short'].sum()} posts as too short.")
    print(f"  Sample:\n{train_df['cleaned_text'].head(3).to_string()}")

    train_df.to_csv("/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/train.csv", index=False)
    print(f" Processed training data: \n")

    # Test data 
    print(f"Loading test data from {args.test_path}...")
    test_df = pd.read_csv(args.test_path)
    print(f"  Shape: {test_df.shape}")

    assert "text" in test_df.columns, "Expected a 'text' column in test data."
    assert "id"   in test_df.columns, "Expected an 'id' column in test data."

    test_df = post_qualitycheck(test_df, "text")

    print(f"  Running TextPreprocessor on test data in batches of {BATCH_SIZE}...")
    test_df['cleaned_text'] = run_in_batches(p, test_df["text"].tolist(), BATCH_SIZE)
    test_df["too_short"]    = test_df['cleaned_text'].apply(too_short_text)

    print(f"  Flagged {test_df['too_short'].sum()} posts as too short.")
    print(f"  Sample:\n{test_df['cleaned_text'].head(3).to_string()}")

    test_df.to_csv("/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/test.csv", index=False)
    print(f" Processed test data:  \n")

    # Summary 
    print("Preprocessing complete for Training and Testing datasets")
    print("\n")
    print(f"  Train : {train_df.shape[0]} posts | "
          f"{train_df['is_image'].sum()} images | "
          f"{train_df['is_removed'].sum()} removed | "
          f"{train_df['too_short'].sum()} too short")
    print(f"  Test  : {test_df.shape[0]} posts | "
          f"{test_df['is_image'].sum()} images | "
          f"{test_df['is_removed'].sum()} removed | "
          f"{test_df['too_short'].sum()} too short")
    print("  ⚠️  Remember to comment out %run run_preprocessing.py in main file")


if __name__ == "__main__":
    main()