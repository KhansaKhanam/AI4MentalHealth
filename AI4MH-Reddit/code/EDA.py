'''
Exploratory Data Analysis (EDA) for AI Mental Health Reddit Study
Analyses the functional label (train) and engagement/temporal patterns (test)

PLOTS:
  1.  functional_class_distribution.png   — label counts + % imbalance
  2.  text_length_train_vs_test.png        — token length histograms side by side
  3.  token_length_by_label.png            — box plots: do longer posts signal sentiment?
  4.  negation_by_label.png               — negation token frequency per label
  5.  ngrams_functional_positive.png      — top bigrams for positive posts
  6.  ngrams_functional_negative.png      — top bigrams for negative posts
  7.  wordclouds_functional.png            — word clouds positive vs negative
  8.  posts_over_time.png                 — volume of Reddit activity over 10 months
  9.  score_distribution.png             — engagement score: posts vs comments
  10. high_engagement_wordcloud.png       — what do viral posts (score>50) talk about?


'''

import os
import re
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from wordcloud import WordCloud 
warnings.filterwarnings("ignore")

# Constants
LABEL_COL = "functional"
 
LABEL_MAP = {
    1:   "positive",
    0:   "neutral",
   -1:   "negative",
    99:  "not_mentioned"
}
 
COLOR_MAP = {
    "positive":      "#2ca02c",
    "neutral":       "#1f77b4",
    "negative":      "#d62728",
    "not_mentioned": "#7f7f7f"
}
 
PLOT_PATH = "/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/Plots"

# Helper Functions for visualization 
def setup():
    os.makedirs(PLOT_PATH, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
    })
 
 
def save(fname):
    path = os.path.join(PLOT_PATH, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")

# Feature Engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["token_count"]    = df["cleaned_text"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
    df["unique_words"]   = df["cleaned_text"].apply(
        lambda x: len(set(str(x).split())) if isinstance(x, str) else 0
    )
    df["negation_count"] = df["cleaned_text"].apply(
        lambda x: len(re.findall(r"not_\w+", str(x)))
    )
    df["label_name"]     = df[LABEL_COL].map(LABEL_MAP) if LABEL_COL in df.columns else None
    return df

# Plot 1: Label distribution 
 
def plot_class_distribution(train_df: pd.DataFrame):
    """
    Bar chart of functional label distribution with percentage labels.
    Highlights the class imbalance visually.
    """
    order  = ["positive", "neutral", "negative", "not_mentioned"]
    counts = train_df["label_name"].value_counts().reindex(order, fill_value=0)
    total  = len(train_df)
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLOR_MAP[l] for l in counts.index],
                  edgecolor="white", linewidth=0.8, width=0.6)
 
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
 
    ax.set_title("Functional Label Distribution — Training Set",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Posts")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.tick_params(axis="x", rotation=10)
 
    # Imbalance annotation
    minority = counts[["positive", "neutral", "negative"]].min()
    majority = counts["not_mentioned"]
    ax.annotate(f"Imbalance ratio: {majority/minority:.1f}x\n(not_mentioned vs rarest class)",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#ffc107"))
 
    plt.tight_layout()
    save("functional_class_distribution.png")
 
    print("\n  Label breakdown:")
    for label, cnt in counts.items():
        print(f"    {label:<18}: {cnt:>4}  ({cnt/total*100:.1f}%)")

# Plot 2: Text length — train vs test 
def plot_text_length(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Side-by-side histograms showing token count distributions.
    Tells you whether train and test posts are similar in length.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Token Count Distribution — Train vs Test",
                 fontsize=13, fontweight="bold")
 
    for ax, df, title, color in zip(
        axes,
        [train_df, test_df],
        [f"Training Set  (n={len(train_df):,})",
         f"Test Set  (n={len(test_df):,})"],
        ["#1f77b4", "#ff7f0e"]
    ):
        ax.hist(df["token_count"], bins=40, color=color,
                edgecolor="white", alpha=0.85)
        median = df["token_count"].median()
        ax.axvline(median, color="black", linestyle="--", linewidth=1.5,
                   label=f"Median: {median:.0f} tokens")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        ax.legend()
 
    plt.tight_layout()
    save("text_length_train_vs_test.png")
 
    print("\n  Token length stats:")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        s = df["token_count"].describe()
        print(f"    {name}  mean={s['mean']:.1f}  "
              f"median={s['50%']:.1f}  "
              f"max={s['max']:.0f}")
 
    return train_df["token_count"].describe()
 
 
# Plot 3: Token length by label 
def plot_length_by_label(train_df: pd.DataFrame):
    """
    Box plots showing token length per label.
    Key insight: do people write more when they have a strong opinion?
    not_mentioned posts tend to be much shorter — they're passing mentions.
    """
    order = ["positive", "neutral", "negative", "not_mentioned"]
    data  = [train_df[train_df["label_name"] == l]["token_count"].dropna()
             for l in order]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
 
    for patch, label in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR_MAP[label])
        patch.set_alpha(0.7)
 
    ax.set_title("Token Count by Functional Label\n",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Token Count")
    ax.set_xlabel("Functional Label")
 
    # Annotate medians
    for i, d in enumerate(data):
        ax.text(i + 1, d.median() + 2, f"{d.median():.0f}",
                ha="center", fontsize=8, color="black", fontweight="bold")
 
    plt.tight_layout()
    save("token_length_by_label.png")
 
    print("\n  Median token count by label:")
    for label, d in zip(order, data):
        print(f"    {label:<18}: {d.median():.0f} tokens  (mean={d.mean():.0f})")
 
 
# Plot 4: Negation frequency by label 
def plot_negation_by_label(train_df: pd.DataFrame):
    """
    Average number of negation tokens (not_X) per post per label.
    Negation handling in preprocessing turns 'not helpful' into 'not_helpful'
    so this directly measures how often each sentiment class uses negation.
    Neutral posts have the highest negation — they're expressing mixed views.
    """
    order  = ["positive", "neutral", "negative", "not_mentioned"]
    means  = train_df.groupby("label_name")["negation_count"].mean().reindex(order)
 
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(means.index, means.values,
                  color=[COLOR_MAP[l] for l in means.index],
                  edgecolor="white", width=0.5)
 
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
 
    ax.set_title("Average Negation Token Count by Label\n"
                 "e.g. 'not_helpful', 'not_mentioned', 'not_replace'",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Negation Tokens per Post")
    ax.set_xlabel("Functional Label")
    ax.tick_params(axis="x", rotation=10)
 
    plt.tight_layout()
    save("negation_by_label.png")
 
    print("\n  Mean negation tokens per post by label:")
    for label, val in means.items():
        print(f"    {label:<18}: {val:.3f}")
 
 
# Plot 5 & 6: N-grams
def plot_ngrams(train_df: pd.DataFrame, top_n: int = 15, ngram_range=(1, 2)):
    """
    Top uni/bigrams for positive and negative posts.
    Reveals the specific language patterns associated with each sentiment.
    """
    for label in ["positive", "negative"]:
        subset = train_df[train_df["label_name"] == label]["cleaned_text"].dropna()
        if len(subset) < 3:
            print(f"  ⚠️  Too few {label} posts for n-gram analysis (n={len(subset)})")
            continue
 
        try:
            vec   = CountVectorizer(ngram_range=ngram_range, max_features=3000,
                                    token_pattern=r"(?u)\b\w\w+\b")
            X     = vec.fit_transform(subset)
            freqs = dict(zip(vec.get_feature_names_out(), X.toarray().sum(axis=0)))
            top   = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
            if not top:
                continue
            terms, counts = zip(*top)
        except ValueError:
            continue
 
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(terms))
        bars  = ax.barh(y_pos, counts, color=COLOR_MAP[label],
                        edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Uni/Bigrams — Functional | {label.title()}\n"
                     f"(n={len(subset)} posts)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")
 
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8)
 
        plt.tight_layout()
        save(f"ngrams_functional_{label}.png")
 
 
# Plot 7: Word clouds
def plot_wordcloud(train_df: pd.DataFrame):
    """
    Word clouds for positive and negative posts.
    Gives an at-a-glance view of dominant vocabulary per sentiment.
    """
    labels_present = [l for l in ["positive", "negative"]
                      if l in train_df["label_name"].values]
    if not labels_present:
        return
 
    fig, axes = plt.subplots(1, len(labels_present),
                              figsize=(8 * len(labels_present), 5))
    if len(labels_present) == 1:
        axes = [axes]
    fig.suptitle("Word Clouds — Functional Factor (Train)",
                 fontsize=13, fontweight="bold")
 
    for ax, label in zip(axes, labels_present):
        corpus = " ".join(
            train_df[train_df["label_name"] == label]["cleaned_text"].dropna()
        )
        if not corpus.strip():
            ax.axis("off")
            continue
 
        wc = WordCloud(width=700, height=450, background_color="white",
                       colormap="Greens" if label == "positive" else "Reds",
                       max_words=80).generate(corpus)
 
        word_freqs = sorted(wc.words_.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  [{label}] Top 10 words:")
        for word, freq in word_freqs:
            print(f"    {word:<22} {'█' * int(freq * 20)}")
 
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label.title()}  (n={len(train_df[train_df['label_name']==label])})",
                     fontsize=11, fontweight="bold", color=COLOR_MAP[label])
        ax.axis("off")
 
    plt.tight_layout()
    save("wordclouds_functional.png")
 
 
# Plot 8: Posts over time
def plot_posts_over_time(test_df: pd.DataFrame):
    """
    Monthly volume of Reddit posts and comments about AI mental health.
    Shows whether public interest is growing, declining, or stable.
    Data spans April 2025 to January 2026.
    """
    df = test_df.copy()
    df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
    df["month"]    = df["datetime"].dt.to_period("M").astype(str)
 
    monthly = df.groupby(["month", "type"]).size().unstack(fill_value=0)
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(monthly))
 
    colors = {"post": "#1f77b4", "comment": "#aec7e8"}
    for col in ["post", "comment"]:
        if col in monthly.columns:
            ax.bar(monthly.index, monthly[col], bottom=bottom,
                   label=col.title() + "s", color=colors.get(col, "#ccc"),
                   edgecolor="white", width=0.6)
            bottom += monthly[col].values
 
    ax.set_title("Reddit Activity Over Time — AI Mental Health\n"
                 "April 2025 – January 2026",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Posts / Comments")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
 
    # Annotate totals
    for i, total in enumerate(bottom):
        ax.text(i, total + 20, f"{int(total):,}",
                ha="center", fontsize=8, fontweight="bold")
 
    plt.tight_layout()
    save("posts_over_time.png")
 
    print("\n  Monthly activity:")
    print(monthly.to_string())
 
 
# Plot 9: Score distribution
def plot_score_distribution(test_df: pd.DataFrame):
    """
    Engagement score distributions for posts vs comments.
    Posts score much higher than comments — a few go viral (score > 50).
    This tells us which content resonated most with the community.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Engagement Score Distribution — Posts vs Comments",
                 fontsize=13, fontweight="bold")
 
    for ax, ptype, color in zip(axes,
                                  ["post", "comment"],
                                  ["#1f77b4", "#ff7f0e"]):
        subset = test_df[test_df["type"] == ptype]["score"]
        # Cap at 95th percentile for readability
        cap = np.percentile(subset, 95)
        ax.hist(subset[subset <= cap], bins=40,
                color=color, edgecolor="white", alpha=0.85)
        ax.axvline(subset.median(), color="black", linestyle="--",
                   label=f"Median: {subset.median():.0f}")
        ax.set_title(f"{ptype.title()}s  (n={len(subset):,})\n"
                     f"max={subset.max()}  mean={subset.mean():.1f}",
                     fontweight="bold")
        ax.set_xlabel("Score (upvotes)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.note = ax.annotate(f"Showing up to 95th percentile (≤{cap:.0f})",
                              xy=(0.98, 0.95), xycoords="axes fraction",
                              ha="right", va="top", fontsize=7,
                              color="grey")
 
    plt.tight_layout()
    save("score_distribution.png")
 
    print("\n  Score stats:")
    print(test_df.groupby("type")["score"].describe().to_string())
 
 
# Plot 10: High engagement word cloud 
def plot_high_engagement_wordcloud(test_df: pd.DataFrame, threshold: int = 50):
    """
    Word cloud of posts that scored above `threshold`.
    These are the posts that resonated most with the community —
    understanding their language reveals what drives engagement.
    """
    high = test_df[test_df["score"] >= threshold]["cleaned_text"].dropna()
    low  = test_df[test_df["score"] < threshold]["cleaned_text"].dropna()
 
    if len(high) < 5:
        print(f"  ⚠️  Only {len(high)} posts above score threshold {threshold}. Skipping.")
        return
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"What Gets Upvoted? High vs Low Engagement Posts\n"
                 f"High = score ≥ {threshold}  ({len(high)} posts)  |  "
                 f"Low = score < {threshold}  ({len(low):,} posts)",
                 fontsize=12, fontweight="bold")
 
    for ax, corpus, title, cmap in zip(
        axes,
        [" ".join(high), " ".join(low)],
        [f"High Engagement (≥{threshold})", "Low Engagement"],
        ["YlOrRd", "Blues"]
    ):
        if not corpus.strip():
            ax.axis("off")
            continue
        wc = WordCloud(width=700, height=420, background_color="white",
                       colormap=cmap, max_words=60).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
 
    plt.tight_layout()
    save("high_engagement_wordcloud.png")
 
    # Print top words in high engagement
    wc_high = WordCloud(max_words=10).generate(" ".join(high))
    top_words = sorted(wc_high.words_.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top words in high-engagement posts (score ≥ {threshold}):")
    for word, freq in top_words:
        print(f"    {word:<22} {'█' * int(freq * 20)}")
 
 
#bSummary report
def write_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, length_stats):
    lines = [
        "=" * 65,
        "EDA SUMMARY — AI Mental Health Reddit Study",
        "Functional Factor",
        "=" * 65,
        f"\nTraining set : {len(train_df):>6,} posts",
        f"Test set     : {len(test_df):>6,} posts",
        f"Date range   : April 2025 – January 2026 (278 days)",
    ]
 
    # Quality flags
    lines.append("\n── Quality Flags ────────────────────────────────────────")
    for col in ["is_removed", "is_image", "too_short"]:
        tr = train_df[col].sum() if col in train_df.columns else "N/A"
        te = test_df[col].sum()  if col in test_df.columns  else "N/A"
        lines.append(f"  {col:<18}  train={tr}   test={te}")
 
    # Token length
    lines.append("\n── Token Length (Train) ─────────────────────────────────")
    if length_stats is not None:
        lines.append(length_stats.to_string())
 
    # Label distribution
    lines.append("\n── Functional Label Distribution (Train) ────────────────")
    counts = train_df["label_name"].value_counts()
    for label, cnt in counts.items():
        pct = cnt / len(train_df) * 100
        lines.append(f"  {label:<18}: {cnt:>4}  ({pct:.1f}%)")
 
    # Imbalance
    non99 = train_df[train_df[LABEL_COL] != 99][LABEL_COL].value_counts()
    if len(non99) > 1:
        ratio = non99.max() / non99.min()
        lines.append(f"\n  Imbalance ratio (excl. not_mentioned): {ratio:.1f}x")
        if ratio > 3:
            lines.append("  ⚠️  Notable imbalance — use class weights in model")
 
    # Test engagement
    lines.append("\n── Test Set Engagement ──────────────────────────────────")
    lines.append(f"  Posts    : {len(test_df[test_df['type']=='post']):,}")
    lines.append(f"  Comments : {len(test_df[test_df['type']=='comment']):,}")
    lines.append(f"  High engagement (score≥50) : {len(test_df[test_df['score']>=50])}")
    lines.append(f"  Median score (posts)    : {test_df[test_df['type']=='post']['score'].median():.0f}")
    lines.append(f"  Median score (comments) : {test_df[test_df['type']=='comment']['score'].median():.0f}")
 
    lines += ["", "=" * 65]
    summary = "\n".join(lines)
 
    save_path = os.path.join(PLOT_PATH, "eda_summary.txt")
    with open(save_path, "w") as f:
        f.write(summary)
    print(f"  ✓ {save_path}")
    print("\n" + summary)
 
 
# Main Function
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/train.csv")
    parser.add_argument("--test",  default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/test.csv")
    args, _ = parser.parse_known_args()
 
    setup()
 
    print(f"\nLoading {args.train}...")
    train_df = pd.read_csv(args.train)
    print(f"  Shape: {train_df.shape}")
 
    print(f"\nLoading {args.test}...")
    test_df = pd.read_csv(args.test)
    print(f"  Shape: {test_df.shape}")
 
    # Feature engineering
    train_df = add_features(train_df)
    test_df["token_count"] = test_df["cleaned_text"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
 
    print("\n[1/10] Label distribution (train)...")
    plot_class_distribution(train_df)
 
    print("\n[2/10] Text length — train vs test...")
    length_stats = plot_text_length(train_df, test_df)
 
    print("\n[3/10] Token length by label (train)...")
    plot_length_by_label(train_df)
 
    print("\n[4/10] Negation frequency by label (train)...")
    plot_negation_by_label(train_df)
 
    print("\n[5/10] N-grams (train)...")
    plot_ngrams(train_df)
 
    print("\n[6/10] Word clouds (train)...")
    plot_wordcloud(train_df)
 
    print("\n[7/10] Posts over time (test)...")
    plot_posts_over_time(test_df)
 
    print("\n[8/10] Score distribution (test)...")
    plot_score_distribution(test_df)
 
    print("\n[9/10] High engagement word cloud (test)...")
    plot_high_engagement_wordcloud(test_df, threshold=50)
 
    print("\n[10/10] Summary report...")
    write_summary(train_df, test_df, length_stats)
 
    print(f"\n✅ EDA complete. All plots saved to {PLOT_PATH}/")
 
 
if __name__ == "__main__":
    main()
