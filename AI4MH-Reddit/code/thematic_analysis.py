"""
Thematic Analysis of Reddit posts related to mental health, using the final predictions from the CatLLM classification. 
This script will generate visualizations and summaries of the themes present in the data, which can provide insights 
into common topics and concerns among Reddit users discussing mental health.

1. Change of Topics over Time: Analyze how the themes in the Reddit posts evolve over time. This can be done by 
categorizing posts into different time periods (e.g., monthly, quarterly) and visualizing the distribution of themes in each period.
2. High-engagement Themes: Identify which themes are associated with higher engagement (e.g., more comments, upvotes). 
This can help understand what topics resonate most with the Reddit community.
3. Per-label word frequency: Analyze the most common words or phrases associated with each predicted label (e.g., "depression", "anxiety").
4. Summary of themes: Provide a summary of the main themes identified in the Reddit posts.
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

# CONSTANTS
LABEL_NAMES   = ["negative", "neutral", "positive", "not_mentioned"]
LABEL_COLORS  = {
    "negative":      "#d62728",
    "neutral":       "#7f7f7f",
    "positive":      "#2ca02c",
    "not_mentioned": "#aec7e8",
}
MIN_TOPIC_SIZE   = 10   # minimum cluster size for BERTopic
N_TOP_WORDS      = 10   # words per topic to display
HIGH_ENG_THRESH  = 50   # score threshold for "high engagement"

# HELPER FUNCTIONS

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Use final_label (post CatLLM review) if available, else functional_label
    if "final_label" in df.columns:
        df["functional_label"] = df["final_label"]
        print("  Using 'final_label' column (post CatLLM review)")
    elif "functional_label" in df.columns:
        print("  Using 'functional_label' column (raw RoBERTa predictions)")
    else:
        raise AssertionError("Missing both 'final_label' and 'functional_label' columns.")

    assert "cleaned_text" in df.columns, "Missing 'cleaned_text' column."

    # Parse timestamp if available
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df["month"]      = df["created_dt"].dt.to_period("M").astype(str)

    # Drop rows with no usable text
    df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.strip() != "")]
    print(f"  Loaded {len(df):,} rows with valid text.")
    return df


def save_fig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ── 1. Label Distribution Overview ───────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, plots_dir: str):
    counts = df["functional_label"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[LABEL_COLORS[l] for l in counts.index], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({val/len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Predicted Label Distribution — Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Posts/Comments")
    ax.set_xlabel("Sentiment Label")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    save_fig(fig, os.path.join(plots_dir, "label_distribution.png"))


# ── 2. BERTopic per Label ─────────────────────────────────────────────────────

def run_bertopic_per_label(df: pd.DataFrame, plots_dir: str, output_dir: str):
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠️  BERTopic or sentence-transformers not installed.")
        print("      Run: pip install bertopic sentence-transformers umap-learn hdbscan")
        return {}

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    all_topic_rows  = []
    topic_models    = {}

    for label in LABEL_NAMES:
        subset = df[df["functional_label"] == label]["cleaned_text"].tolist()
        print(f"\n  [{label}]  {len(subset):,} posts")

        if len(subset) < MIN_TOPIC_SIZE * 2:
            print(f"    ⚠️  Too few posts for topic modelling (need ≥ {MIN_TOPIC_SIZE*2}). Skipping.")
            continue

        topic_model = BERTopic(
            embedding_model    = embedding_model,
            min_topic_size     = MIN_TOPIC_SIZE,
            calculate_probabilities = False,
            verbose            = False,
        )
        topics, _ = topic_model.fit_transform(subset)
        topic_models[label] = topic_model

        # Topic info table
        info = topic_model.get_topic_info()
        info["label"] = label
        all_topic_rows.append(info)

        n_topics = len(info[info["Topic"] != -1])
        print(f"    Found {n_topics} topics  |  "
              f"Outliers: {(np.array(topics) == -1).sum():,}")

        # ── Bar chart of top topics ──────────────────────────────────────────
        top_topics = info[info["Topic"] != -1].head(8)
        if len(top_topics) == 0:
            continue

        topic_labels = []
        topic_counts = []
        for _, row in top_topics.iterrows():
            tid   = row["Topic"]
            words = topic_model.get_topic(tid)
            if words:
                label_str = " | ".join([w for w, _ in words[:3]])
            else:
                label_str = f"Topic {tid}"
            topic_labels.append(f"T{tid}: {label_str}")
            topic_counts.append(row["Count"])

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(topic_labels[::-1], topic_counts[::-1],
                       color=LABEL_COLORS[label], edgecolor="white")
        for bar, val in zip(bars, topic_counts[::-1]):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8)
        ax.set_title(f"Top Topics — {label.capitalize()} Posts", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Posts")
        sns.despine()
        save_fig(fig, os.path.join(plots_dir, f"topics_{label}.png"))

        # ── Top words per top 3 topics ───────────────────────────────────────
        top3 = info[info["Topic"] != -1].head(3)["Topic"].tolist()
        n_cols = len(top3)
        if n_cols == 0:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for ax, tid in zip(axes, top3):
            words = topic_model.get_topic(tid)
            if not words:
                continue
            wds   = [w for w, _ in words[:N_TOP_WORDS]]
            scores = [s for _, s in words[:N_TOP_WORDS]]
            ax.barh(wds[::-1], scores[::-1], color=LABEL_COLORS[label], edgecolor="white")
            ax.set_title(f"T{tid} Top Words", fontsize=10, fontweight="bold")
            ax.set_xlabel("Relevance Score")
            sns.despine()

        fig.suptitle(f"Top Words in Leading Topics — {label.capitalize()}",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_fig(fig, os.path.join(plots_dir, f"topic_words_{label}.png"))

    # ── Save combined topic table ────────────────────────────────────────────
    if all_topic_rows:
        combined = pd.concat(all_topic_rows, ignore_index=True)
        combined = combined[combined["Topic"] != -1]  # drop outlier row
        out_path = os.path.join(output_dir, "all_topics.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n  Topic summary saved → {out_path}")

    return topic_models


# ── 3. Topics Over Time ───────────────────────────────────────────────────────

def plot_topics_over_time(df: pd.DataFrame, topic_models: dict, plots_dir: str):
    if "month" not in df.columns:
        print("  ⚠️  No 'created_utc' column — skipping topics over time.")
        return

    for label in LABEL_NAMES:
        if label not in topic_models:
            print(f"  ⚠️  No topic model for '{label}' — skipping.")
            continue

        subset = df[df["functional_label"] == label][["cleaned_text", "month"]].dropna()
        if len(subset) < 20:
            print(f"  ⚠️  Too few '{label}' posts for topics over time — skipping.")
            continue

        model     = topic_models[label]
        docs      = subset["cleaned_text"].tolist()
        months    = subset["month"].tolist()
        topics, _ = model.transform(docs)

        topics_over_time = model.topics_over_time(docs, months, nr_bins=10)

        top_topic_ids = (
            topics_over_time.groupby("Topic")["Frequency"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        top_topic_ids = [t for t in top_topic_ids if t != -1]

        if not top_topic_ids:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        for tid in top_topic_ids:
            t_data = topics_over_time[topics_over_time["Topic"] == tid].sort_values("Timestamp")
            words  = model.get_topic(tid)
            lbl    = " | ".join([w for w, _ in words[:3]]) if words else f"Topic {tid}"
            ax.plot(t_data["Timestamp"], t_data["Frequency"],
                    marker="o", label=f"T{tid}: {lbl}")

        ax.set_title(f"Top Topics Over Time — {label.capitalize()} Posts",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.xticks(rotation=45)
        sns.despine()
        plt.tight_layout()
        save_fig(fig, os.path.join(plots_dir, f"topics_over_time_{label}.png"))


# ── 4. High Engagement Topic Analysis ────────────────────────────────────────

def plot_high_engagement_topics(df: pd.DataFrame, topic_models: dict, plots_dir: str):
    if "score" not in df.columns:
        print("  ⚠️  No 'score' column — skipping high-engagement analysis.")
        return

    high_eng = df[df["score"] >= HIGH_ENG_THRESH]
    if len(high_eng) == 0:
        print(f"  ⚠️  No posts with score ≥ {HIGH_ENG_THRESH}.")
        return

    print(f"\n  High-engagement posts (score ≥ {HIGH_ENG_THRESH}): {len(high_eng):,}")

    label_eng = high_eng["functional_label"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    fig, ax   = plt.subplots(figsize=(8, 5))
    ax.bar(label_eng.index, label_eng.values,
           color=[LABEL_COLORS[l] for l in label_eng.index], edgecolor="white")
    ax.set_title(f"Label Distribution — High Engagement Posts (score ≥ {HIGH_ENG_THRESH})",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    sns.despine()
    save_fig(fig, os.path.join(plots_dir, "high_engagement_label_dist.png"))

    # Word frequency in high-engagement posts per label
    for label in LABEL_NAMES:
        subset = high_eng[high_eng["functional_label"] == label]["cleaned_text"].dropna()
        if len(subset) < 5:
            continue

        all_words = " ".join(subset.tolist()).split()
        top_words = Counter(all_words).most_common(15)
        words, counts = zip(*top_words)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(list(words)[::-1], list(counts)[::-1],
                color=LABEL_COLORS[label], edgecolor="white")
        ax.set_title(f"Top Words in High-Engagement {label.capitalize()} Posts",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")
        sns.despine()
        save_fig(fig, os.path.join(plots_dir, f"high_eng_words_{label}.png"))


# ── 5. Sentiment Share Over Time ──────────────────────────────────────────────

def plot_sentiment_over_time(df: pd.DataFrame, plots_dir: str):
    if "month" not in df.columns:
        return

    monthly = (
        df.groupby(["month", "functional_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABEL_NAMES, fill_value=0)
    )
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom  = np.zeros(len(monthly_pct))
    for label in LABEL_NAMES:
        if label in monthly_pct.columns:
            ax.bar(monthly_pct.index, monthly_pct[label],
                   bottom=bottom, label=label,
                   color=LABEL_COLORS[label], edgecolor="white", width=0.6)
            bottom += monthly_pct[label].values

    ax.set_title("Sentiment Share Over Time (Monthly)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Posts (%)")
    ax.set_xlabel("Month")
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=45)
    sns.despine()
    plt.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "sentiment_over_time.png"))


# ── 6. Word Frequency per Label ───────────────────────────────────────────────

def plot_word_frequency_per_label(df: pd.DataFrame, plots_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, label in zip(axes, LABEL_NAMES):
        subset    = df[df["functional_label"] == label]["cleaned_text"].dropna()
        all_words = " ".join(subset.tolist()).split()
        top_words = Counter(all_words).most_common(15)

        if not top_words:
            ax.set_visible(False)
            continue

        words, counts = zip(*top_words)
        ax.barh(list(words)[::-1], list(counts)[::-1],
                color=LABEL_COLORS[label], edgecolor="white")
        ax.set_title(f"{label.capitalize()}  (n={len(subset):,})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")
        sns.despine(ax=ax)

    fig.suptitle("Top 15 Words per Sentiment Label", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, os.path.join(plots_dir, "word_frequency_per_label.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Thematic analysis on classified Reddit posts.")
    parser.add_argument("--input",  default="processed_input/test_finalPreds.csv",
                        help="Path to classified predictions CSV")
    parser.add_argument("--plots",  default="Plots/",
                        help="Directory to save all plots")
    parser.add_argument("--output", default="thematic_output/",
                        help="Directory to save topic tables and summaries")
    args = parser.parse_args()

    os.makedirs(args.plots,  exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Thematic Analysis — AI Mental Health Reddit")
    print(f"{'='*60}")

    print(f"\n[1/6] Loading data from {args.input}...")
    df = load_data(args.input)

    print("\n[2/6] Plotting label distribution...")
    plot_label_distribution(df, args.plots)

    print("\n[3/6] Plotting word frequency per label...")
    plot_word_frequency_per_label(df, args.plots)

    print("\n[4/6] Plotting sentiment share over time...")
    plot_sentiment_over_time(df, args.plots)

    print("\n[5/6] Running BERTopic per sentiment label...")
    topic_models = run_bertopic_per_label(df, args.plots, args.output)

    print(f"\n[6/6] Topics over time for all labels...")
    plot_topics_over_time(df, topic_models, args.plots)

    print("\n[+] High-engagement topic analysis...")
    plot_high_engagement_topics(df, topic_models, args.plots)

    print(f"\n{'='*60}")
    print("  Thematic analysis complete.")
    print(f"  Plots   → {args.plots}")
    print(f"  Topics  → {args.output}all_topics.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

