'''
Takes low_conf_review.csv (rows where MentalRoBERTa confidence < 0.40)
and re-classifies them using CatLLM (LLM-backed classification).

The two predictions are then merged back into the full test_withPred.csv,
with low-confidence rows replaced by the CatLLM prediction.
'''

import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
from groq import Groq

# CONSTANTS 

CATEGORIES = [
    "positive  — the post views AI as beneficial, helpful, or supportive for mental health",
    "negative  — the post views AI as harmful, ineffective, or inappropriate for mental health",
    "neutral   — the post is balanced, ambiguous, or neither positive nor negative about AI",
    "not_mentioned — AI's functional role in mental health is not discussed in this post",
]

LABEL2RAW = {
    "positive":1,
    "negative":-1,
    "neutral":0,
    "not_mentioned":99,
}

# HELPER FUNCTIONS

def normalise_label(label: str) -> str:
    if not isinstance(label, str):
        return "not_mentioned"
    
    clean = label.split("-")[0].strip().lower()
    for key in LABEL2RAW:
        if key in clean:
            return key
    return "not_mentioned"

def run(texts: pd.Series, model: str, api_key: str,
        checkpoint_path: str = "groq_checkpoint.csv") -> pd.Series:

    client = Groq(api_key=api_key)

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        done      = pd.read_csv(checkpoint_path)
        labels    = done["label"].tolist()
        start_idx = len(labels)
        print(f"  Resuming from row {start_idx} ({start_idx} rows already done)")
    else:
        labels    = []
        start_idx = 0

    category_prompt = """You are a research assistant classifying Reddit posts about AI and mental health.

Classify the post into exactly one of these four categories:
- positive: the post views AI as beneficial, helpful, or supportive for mental health
- negative: the post views AI as harmful, ineffective, or inappropriate for mental health
- neutral: the post is balanced, ambiguous, or neither positive nor negative about AI
- not_mentioned: AI's functional role in mental health is not discussed in this post

Reply with ONLY the single category word. Nothing else."""

    for i, text in enumerate(texts[start_idx:], start=start_idx):
        try:
            response = client.chat.completions.create(
                model      = "llama-3.1-8b-instant",
                max_tokens = 10,
                messages   = [{"role": "user", "content": f"{category_prompt}\n\nPost:\n{str(text)[:2000]}"}]
            )
            label = response.choices[0].message.content.strip().lower()
            labels.append(normalise_label(label))

        except Exception as e:
            if "429" in str(e):
                print(f"\n  Daily limit hit at row {i}. Saving checkpoint and stopping.")
                pd.DataFrame({"label": labels}).to_csv(checkpoint_path, index=False)
                print(f"  Checkpoint saved → {checkpoint_path}  ({len(labels)} rows done)")
                print(f"  Re-run tomorrow — it will automatically resume from row {len(labels)}")
                break
            else:
                print(f"  Row {i} failed: {e}")
                labels.append("not_mentioned")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(texts)} rows...")
            pd.DataFrame({"label": labels}).to_csv(checkpoint_path, index=False)

        time.sleep(0.1)

    # Pad remaining rows with not_mentioned if stopped early
    while len(labels) < len(texts):
        labels.append("not_mentioned")

    pd.DataFrame({"label": labels}).to_csv(checkpoint_path, index=False)
    return pd.Series(labels)

def label_agreement(df: pd.DataFrame):
    combined_df = df[df["roberta_pred"].notna() &
                     df["catllm_label"].notna()].copy()
    
    if len(combined_df) == 0:
        return 
    
    combined_df["agree"] = (combined_df["roberta_label"] == combined_df["catllm_label"])
    agreement_percent = (combined_df["agree"].mean())*100

    print("\n Agreement between mental/mental-roberta-base and CatLLM:")
    print(f" Total Compared: {len(combined_df):,}")
    print(f" Agree : {combined_df['agree'].sum():,} ({agreement_percent:.1f}%)")
    print(f" Disgree : {(~combined_df['agree']).sum():,}  ({100-agreement_percent:.1f}%)")

    print(f"\n Disagreement Breakdown: ")
    disagree = combined_df[~combined_df["agree"]][["roberta_label", "catllm_label"]]
    print(disagree.value_counts().head(10).to_string())


# MAIN FUNCTION

def main():
    parser = argparse.ArgumentParser(description="Re-classify low-confidence RoBERTa rows using CatLLM.")
    parser.add_argument("--predictions",
                        default = "data/processed_data/test_withPred.csv",)
    parser.add_argument("--low_conf",
                        default = "data/processed_data/low_conf_review.csv")
    parser.add_argument("--output_dir",
                        default= "data/processed_data/")
    parser.add_argument("--llm_model",
                        default="gemini-1.5-flash")
    parser.add_argument("--groq_key", default="")
    args = parser.parse_args()

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("No API Key provided")

    os.makedirs(args.output_dir, exist_ok = True)


    print("\n CatLLM Review \n")
    print(f"\n Loading Full Predictions -> {args.predictions}")
    full_df = pd.read_csv(args.predictions)
    print(f" Shape : {full_df.shape}")    

    print(f"  Loading low-conf subset -> {args.low_conf}")
    low_df = pd.read_csv(args.low_conf)
    print(f" Shape: {low_df.shape}  ({len(low_df)/len(full_df):.1%} of test set)")

    assert "cleaned_text"      in low_df.columns, "Missing 'cleaned_text' column."
    assert "functional_label"  in low_df.columns, "Missing 'functional_label' column."
    assert "id"                in low_df.columns, "Missing 'id' column."

    low_df = low_df.copy()
    low_df["roberta_label"] = low_df["functional_label"]
    low_df["roberta_pred"]  = low_df["functional_pred"]

    print(f"\n  Running CatLLM ({args.llm_model}) on {len(low_df):,} low-confidence rows...")
    texts = low_df["cleaned_text"].fillna("").reset_index(drop=True)

    catllm_labels = run(texts, model=args.llm_model, api_key=api_key,
                    checkpoint_path=os.path.join(args.output_dir, "groq_checkpoint.csv"))
    catllm_labels.index = low_df.index

    low_df["catllm_label"] = catllm_labels
    low_df["catllm_pred"]  = low_df["catllm_label"].map(LABEL2RAW)

    print(f"\n  CatLLM predicted distribution:")
    print(low_df["catllm_label"].value_counts().to_string())


    label_agreement(low_df)

    audit_path = os.path.join(args.output_dir, "catllm_review.csv")
    low_df.to_csv(audit_path, index=False)
    print(f"Audit File saved to {audit_path}")

    print(f"\n  Merging CatLLM predictions into full dataset...")

    catllm_lookup = low_df.set_index("id")[["catllm_label","catllm_pred"]]
    full_df = full_df.set_index("id")
    full_df["catllm_label"] = catllm_lookup["catllm_label"]
    full_df["catllm_pred"]  = catllm_lookup["catllm_pred"]

    # use CatLLM label whereever available, else roBERTa label
    full_df["final_label"] = full_df["catllm_label"].combine_first(full_df["functional_label"])
    full_df["final_pred"] = full_df["catllm_pred"].combine_first(full_df["functional_pred"])
    full_df = full_df.reset_index()

    print(f"\n  Final label distribution (merged):")
    print(full_df["final_label"].value_counts().to_string())

    final_path = os.path.join(args.output_dir, "test_finalPreds.csv")
    full_df.to_csv(final_path, index=False)
    print(f" Final predictions saved to {args.output_dir}")
    print(f"  Audit (both predictions) saved to {audit_path}")
    print(f"  Final merged predictions saved to {final_path}")

if __name__ == "__main__":
    main()

