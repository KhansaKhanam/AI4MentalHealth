'''
Annotation Agreement: Comparing Human Annotation(manual coding) and CatLLM classification
to restablish how reliable CatLLM classification is

Model used: Groq/Llama
Metrics used for comparison:
    1. Raw Agreement %
    2. Cohen's Kappa
    3. Krippendorff's Alpha

Expected Output:
    1. catllm_traindf.csv (saved to /data/processed_input/)
    2. agreementreport_summary.txt (saved to /data/output_dir/)
    3. agreement_confusion.png (saved to /Plots/)
'''

import os
import re
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from lowconf_data_review import run

try:
    import krippendorff
    KRIPP = True
except ImportError:
    KRIPP = False
    print("Krippendorff's alpha will not be calculated. Please install the 'krippendorff' package to enable this feature.")


from groq import Groq

# CONSTANTS
LABELS = ["negative", "neutral", "positive","not_mentioned"]
RAW2LABEL = {-1:"negative", 0:"neutral", 1:"positive", 99:"not_mentioned"}
LABEL2RAW = {"negative":-1, "neutral":0, "positive":1, "not_mentioned":99}

# HELPER FUNCTIONS
def normalize_label(label):
    if pd.isna(label):
        return None
    if isinstance(label,str):
        label = label.strip().lower()
        for name in LABELS:
            if name in label:
                return name
        return None
    try:
        return RAW2LABEL.get(int(float(label)), None)
    except:
        return None
    
def interpret(score):
    if score is None:
        return "N/A"
    elif score < 0.00:
        return "Poor"
    elif score < 0.20:
        return "Slight"
    elif score < 0.40:
        return "Fair"
    elif score < 0.60:
        return "Moderate"
    elif score < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"

def compute_agreement(human: pd.Series, catllm: pd.Series) -> dict:
    valid = human.notna() & catllm.notna()
    h = human[valid].reset_index(drop=True)
    c = catllm[valid].reset_index(drop=True)
    n = len(h)

    print(f"\n  Total valid samples for agreement: {n:,}")

    if n < 2:
        print("  Not enough valid samples to compute agreement.")
        return {"raw_agreement": None, "cohen_kappa": None, "krippendorff_alpha": None} 
    
    raw_agreement = (h == c).mean()
    cohen_kappa = cohen_kappa_score(h.map(LABEL2RAW), c.map(LABEL2RAW))

    alpha = None
    if KRIPP:
        reliabiltiy = [
            [LABEL2RAW.get(v, np.nan) for v in h.tolist()],
            [LABEL2RAW.get(v, np.nan) for v in c.tolist()]
        ]
        alpha = krippendorff.alpha(reliabiltiy, level_of_measurement="nominal")

    print("\n  Human vs CatLLM Agreement:")    
    print(f"\n N (valid samples): {n:,}")
    print(f"\n  Raw Agreement: {raw_agreement:.4f} ({interpret(raw_agreement)})")
    print(f"\n  Cohen's Kappa: {cohen_kappa:.4f} ({interpret(cohen_kappa)})")
    if alpha is not None:
        print(f"\n  Krippendorff's Alpha: {alpha:.4f} ({interpret(alpha)})")

    print(f"\n Disagreement Breakdown: ")
    print(f"\n {"Label":<15} {'Human':<10} {'CatLLM':<10} {'Agreement':<10}")

    for label in LABELS:
        h_count = (h == label).sum()
        c_count = (c == label).sum()
        agree_count = ((h == label) & (c == label)).sum()
        agreement_rate = agree_count / max(h_count, 1)
        print(f"\n {label:<15} {h_count:<10} {c_count:<10} {agreement_rate:.2%}")

    return {
        "n": n,
        "raw_agreement": round(raw_agreement, 4),
        "cohen_kappa": round(cohen_kappa, 4),
        "kappa_interpretation": interpret(cohen_kappa),
        "krippendorff_alpha": round(alpha, 4) if alpha is not None else None,
        "alpha_interpretation": interpret(alpha) if alpha is not None else None,
    }  

def plot_confusion(human: pd.Series, catllm: pd.Series, save_path: str):
    valid = human.notna() & catllm.notna()
    h = human[valid]
    c = catllm[valid]

    cm  = confusion_matrix(h, c, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("CatLLM")
    ax.set_ylabel("Human")
    ax.set_title("Agreement Matrix — Human vs CatLLM",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


def classify_with_groq(texts: pd.Series,
                       api_key: str,
                       checkpoint_path: str) -> pd.Series:

    client = Groq(api_key=api_key)

    # Fail fast
    print("  Testing Groq API key...")
    try:
        test = client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            max_tokens = 5,
            messages   = [{"role": "user", "content": "Reply with: ok"}]
        )
        print(f"  ✓ Key valid. Response: {test.choices[0].message.content.strip()}")
    except Exception as e:
        raise RuntimeError(f"Key test failed — stopping.\n  Error: {e}")

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        done      = pd.read_csv(checkpoint_path)
        labels    = done["catllm_label"].tolist()
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
            labels.append(normalize_label(label) or "not_mentioned")

        except Exception as e:
            if "429" in str(e):
                wait_match = re.search(r"try again in (\d+)m", str(e))
                wait_mins  = int(wait_match.group(1)) + 1 if wait_match else 2
                print(f"\n  Daily limit hit at row {i}. Saving checkpoint and stopping.")
                pd.DataFrame({"catllm_label": labels}).to_csv(checkpoint_path, index=False)
                print(f"  Checkpoint saved → {checkpoint_path}  ({len(labels)} rows done)")
                print(f"  Re-run tomorrow — it will resume from row {len(labels)}")
                break
            else:
                print(f"  Row {i} failed: {e}")
                labels.append("not_mentioned")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(texts)} rows...")
            pd.DataFrame({"catllm_label": labels}).to_csv(checkpoint_path, index=False)

        time.sleep(0.1)

    # Pad if stopped early
    while len(labels) < len(texts):
        labels.append(None)

    pd.DataFrame({"catllm_label": labels}).to_csv(checkpoint_path, index=False)
    return pd.Series(labels)


# MAIN FUNCTION 

def main():
    parser = argparse.ArgumentParser(
        description="Annotation agreement between train.csv human labels and CatLLM."
    )
    parser.add_argument("--train",    required=True,
                        help="Path to train.csv")
    parser.add_argument("--output",   default="agreement_output/",
                        help="Directory to save all outputs")
    parser.add_argument("--groq_key", default="",
                        help="Groq API key (gsk_...)")
    args = parser.parse_args()

    api_key = args.groq_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("No Groq API key. Use --groq_key or set GROQ_API_KEY.")

    os.makedirs(args.output, exist_ok=True)
    checkpoint_path = os.path.join(args.output, "catllm_train_checkpoint.csv")

    print("\n")
    print("  Annotation Agreement: Human vs CatLLM")
    print("\n")

    print(f"  Loading training data from {args.train}...")
    train_df = pd.read_csv(args.train)
    print(f"  Total training samples: {len(train_df):,} ({train_df.shape[0]} rows, {train_df.shape[1]} columns)")

    assert "text"       in train_df.columns, "Missing 'text' column in train.csv"
    assert "functional" in train_df.columns, "Missing 'functional' column in train.csv"

    train_df["human_label"] = train_df["functional"].apply(normalize_label)
    train_df = train_df[
        train_df["text"].notna() &
        train_df["human_label"].notna()
    ].reset_index(drop=True)

    print(f"\n  Human label distribution:")
    print(train_df["human_label"].value_counts().to_string())

    texts = train_df["text"].fillna("").reset_index(drop=True)
    catllm_labels = classify_with_groq(texts, api_key=api_key,
                                       checkpoint_path=checkpoint_path)
    train_df["catllm_label"] = catllm_labels.values

    print(f"\n  CatLLM label distribution:")
    print(train_df["catllm_label"].value_counts().to_string())

    out_csv = os.path.join(args.output, "catllm_traindf.csv")
    train_df[["text", "functional", "human_label", "catllm_label"]].to_csv(
        out_csv, index=False
    )
    print(f"\n  Labelled file saved → {out_csv}")

    results = compute_agreement(train_df["human_label"], train_df["catllm_label"])
    cm_path = os.path.join(args.output, "agreement_confusion.png")
    plot_confusion(train_df["human_label"], train_df["catllm_label"], save_path=cm_path)

    report_path = os.path.join(args.output, "agreementreport_summary.txt")
    with open(report_path, "w") as f:
        f.write("Annotation Agreement Report: Human vs CatLLM\n")
        f.write("\n")
        f.write(f"Total valid samples: {results['n']}\n")
        f.write(f"Raw Agreement: {results['raw_agreement']} ({interpret(results['raw_agreement'])})\n")
        f.write(f"Cohen's Kappa: {results['cohen_kappa']} ({results['kappa_interpretation']})\n")
        if results["krippendorff_alpha"] is not None:
            f.write(f"Krippendorff's Alpha: {results['krippendorff_alpha']} ({results['alpha_interpretation']})\n")

        f.write("Interpretation guide:\n")
        f.write("  < 0.00        → Poor\n")
        f.write("  0.00 – 0.20   → Slight\n")
        f.write("  0.21 – 0.40   → Fair\n")
        f.write("  0.41 – 0.60   → Moderate\n")
        f.write("  0.61 – 0.80   → Substantial\n")
        f.write("  0.81 – 1.00   → Almost Perfect\n\n")
        for k, v in results.items():
            f.write(f"  {k:<28} : {v}\n")

            
    print(f"  Agreement report saved → {report_path}")
    print(f"  Confusion matrix saved → {cm_path}")
    print(f"\n  All outputs saved to {args.output}")


if __name__ == "__main__":
    main()










