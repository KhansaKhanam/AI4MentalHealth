# AI4MH — Studying Public Perceptions of AI in Mental Health Counseling

> An end-to-end NLP pipeline that collects, preprocesses, classifies, and thematically analyses 20,000+ Reddit posts to understand how the public perceives AI tools in mental health support.

---

## Overview

This project studies the **functional sentiment** expressed in Reddit posts about AI and mental health — asking not just whether people feel positively or negatively, but specifically how they feel about AI's *practical usefulness* as a mental health tool.

The pipeline spans the full research lifecycle: data collection → text preprocessing → human annotation → automated classification → annotation agreement validation → low-confidence review → thematic analysis.

**Data:** ~800 manually annotated Reddit posts (train) + ~19,240 unannotated posts (test)  
**Time span:** April 2025 – January 2026  
**Label scheme:** `positive` | `neutral` | `negative` | `not_mentioned`

---

## Research Questions

1. How does the Reddit community functionally perceive AI tools in mental health contexts?
2. Does sentiment toward AI mental health tools change over time?
3. What themes characterise positive vs negative perceptions?
4. Can a hybrid human + LLM annotation pipeline reliably classify nuanced mental health discourse?

---

## Project Structure

```
AI4MH-Reddit/
│
├── code/
│   ├── preprocessing.py          # TextPreprocessor class
│   ├── run_preprocessing.py      # Run once to clean train + test CSVs
│   ├── EDA.py                    # 10 exploratory visualisations
│   ├── classification.py         # MentalRoBERTa fine-tuning + prediction
│   ├── lowconf_data_review.py    # CatLLM re-classification of low-confidence rows
│   ├── annotation_agreement.py   # Human vs CatLLM agreement metrics
│   └── thematic_analysis.py      # BERTopic + sentiment-over-time analysis
│
├── data/
│   └── processed_input/
│       ├── train.csv             # 801 manually annotated posts
│       ├── test.csv              # 19,240 Reddit posts to classify
│       ├── test_withPred.csv     # RoBERTa predictions
│       ├── low_conf_review.csv   # Posts flagged for CatLLM review
│       ├── catllm_review.csv     # CatLLM predictions on low-conf rows
│       └── test_finalPreds.csv   # Final merged predictions
│
├── Plots/                        # All output visualisations
├── model_output/                 # Saved fine-tuned RoBERTa model
├── thematic_output/              # BERTopic topic tables
├── agreement_output/             # Annotation agreement reports
│
└── mentalhealth.ipynb            # Main control notebook
```

---

## Pipeline

```
train.csv (801 labelled posts)
test.csv  (19,240 unlabelled posts)
        │
        ▼
[1] run_preprocessing.py
    TextPreprocessor:
    emoji → chatwords → contractions → clean → tokenize → negation → stopwords → lemmatize
    Output: cleaned_text column added to both CSVs
        │
        ▼
[2] EDA.py
    10 visualisations: label distribution, token length, negation frequency,
    n-grams, word clouds, posts over time, engagement score analysis
        │
        ▼
[3] classification.py
    Fine-tune mental/mental-roberta-base (10 epochs, class weights)
    Output: test_withPred.csv + low_conf_review.csv (confidence < 0.40)
        │
        ▼
[4] lowconf_data_review.py
    Re-classify uncertain rows with CatLLM (Groq/Llama 3.1)
    Merge back into full predictions
    Output: test_finalPreds.csv
        │
        ▼
[5] annotation_agreement.py
    Run CatLLM on human-labelled train set
    Compute Raw Agreement, Cohen's Kappa, Krippendorff's Alpha
    Output: agreementreport_summary.txt
        │
        ▼
[6] thematic_analysis.py
    BERTopic per sentiment label
    Sentiment share over time (monthly)
    High-engagement topic analysis
    Output: all_topics.csv + visualisations
```

---

## Setup

### Requirements

```bash
pip install pandas openpyxl torch transformers scikit-learn matplotlib seaborn wandb
pip install bertopic sentence-transformers wordcloud umap-learn hdbscan
pip install nltk beautifulsoup4 contractions emoji pyspellchecker langdetect
pip install groq krippendorff
```

### API Keys Required

| Key | Used for | Where to get |
|---|---|---|
| HuggingFace token | Load `mental/mental-roberta-base` | huggingface.co/settings/tokens |
| Groq API key | CatLLM classification (Llama 3.1) | console.groq.com |

---

## Running the Pipeline

### Step 0 — Set paths

In `mentalhealth.ipynb` update:
```python
BASEPATH  = "/path/to/AI4MH-Reddit"
TRAIN_CSV = os.path.join(BASEPATH, "data/processed_input/train.csv")
TEST_CSV  = os.path.join(BASEPATH, "data/processed_input/test.csv")
```

### Step 1 — Preprocessing (run once)

```bash
python run_preprocessing.py \
    --train_path data/processed_input/train.csv \
    --test_path  data/processed_input/test.csv
```

> After first run, comment out the `%run` cell in the notebook.

### Step 2 — EDA

```bash
python EDA.py \
    --train data/processed_input/train.csv \
    --test  data/processed_input/test.csv
```

### Step 3 — Classification

```bash
python classification.py \
    --train      data/processed_input/train.csv \
    --test       data/processed_input/test.csv \
    --output_dir model_output/ \
    --data_dir   data/processed_input/ \
    --epochs     10 \
    --hf_token   YOUR_HF_TOKEN
```

> GPU strongly recommended. Use Google Colab (Runtime → Change runtime type → T4 GPU).

### Step 4 — Low-confidence review

```bash
python lowconf_data_review.py \
    --predictions data/processed_input/test_withPred.csv \
    --low_conf    data/processed_input/low_conf_review.csv \
    --output_dir  data/processed_input/ \
    --groq_key    YOUR_GROQ_KEY
```

> Supports automatic checkpointing — if Groq's daily limit is hit, progress is saved and resumes on the next run.

### Step 5 — Annotation agreement

```bash
python annotation_agreement.py \
    --train    data/processed_input/train.csv \
    --output   agreement_output/ \
    --groq_key YOUR_GROQ_KEY
```

### Step 6 — Thematic analysis

```bash
python thematic_analysis.py \
    --input  data/processed_input/test_finalPreds.csv \
    --plots  Plots/ \
    --output thematic_output/
```

---

## Model

**Base model:** [`mental/mental-roberta-base`](https://huggingface.co/mental/mental-roberta-base)  
RoBERTa-base further pre-trained on mental health Reddit posts.

**Fine-tuning configuration:**

| Parameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 16 |
| Epochs | 10 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Validation split | 20% stratified |
| Confidence threshold | 0.40 |
| Class weights | Inverse frequency |

---

## Label Scheme

| Label | Code | Meaning |
|---|---|---|
| positive | 1 | AI viewed as beneficial, effective, or helpful |
| neutral | 0 | Balanced, mixed, or ambiguous view of AI |
| negative | -1 | AI viewed as harmful, ineffective, or inappropriate |
| not_mentioned | 99 | AI's functional role not discussed in the post |

---

## Annotation Agreement

Human annotations are compared against CatLLM predictions using three metrics:

| Metric | What it measures |
|---|---|
| Raw Agreement % | Percentage of matching labels |
| Cohen's Kappa | Agreement corrected for chance |
| Krippendorff's Alpha | Reliability across raters |

```
Interpretation:
  < 0.00        → Poor
  0.00 – 0.20   → Slight
  0.21 – 0.40   → Fair
  0.41 – 0.60   → Moderate
  0.61 – 0.80   → Substantial
  0.81 – 1.00   → Almost Perfect
```

---

## Known Limitations

**Class imbalance:** `not_mentioned` dominates (~62% of training data). The model performs well on `not_mentioned` and `positive` but struggles with `neutral` and `negative` (~55 examples each).

**Small training set:** 801 labelled posts is at the low end for fine-tuning a transformer. Expanding to 2,000+ examples per class would substantially improve minority class performance.

**CPU training:** Full training takes several hours on CPU. A GPU is strongly recommended.

**Groq daily limits:** The free tier has daily token limits. Checkpointing is built in to handle interruptions automatically.

---

## TextPreprocessor

Custom NLP preprocessing class built for Reddit/social media text.

**Pipeline order:**
```
emoji handling → chat word expansion → contraction expansion →
text cleaning → tokenization → negation handling →
stopword removal → lemmatization
```

**Key design decisions:**
- Reddit artifacts handled: `[removed]`, `[deleted]`, `u/username`, `r/subreddit`
- Mojibake correction (UTF-8 read as Latin-1 produces `â€¦` etc.)
- `_x000D_` removal (Excel/CSV carriage return artifact)
- Negation preservation: `"not helpful"` → `"not_helpful"` so stopword removal does not flip sentiment
- Chat word expansion: `"LOL"` → `"Laughing Out Loud"`
- Negation words excluded from stopwords set so they are never silently dropped

---

## Outputs Summary

| File | Description |
|---|---|
| `test_withPred.csv` | RoBERTa predictions + confidence scores |
| `low_conf_review.csv` | Posts below confidence threshold |
| `test_finalPreds.csv` | Final merged predictions (RoBERTa + CatLLM) |
| `classification_report.txt` | Precision / recall / F1 on validation set |
| `confusion_matrix.png` | Confusion matrix on validation set |
| `agreementreport_summary.txt` | Human vs CatLLM agreement report |
| `agreement_confusion.png` | Agreement confusion matrix |
| `all_topics.csv` | BERTopic topic summary across all labels |
| `Plots/` | All EDA and thematic analysis visualisations |

---

## Acknowledgements

- [`mental/mental-roberta-base`](https://huggingface.co/mental/mental-roberta-base) — Ji et al., domain-adapted RoBERTa for mental health text
- [`BERTopic`](https://maartengr.github.io/BERTopic/) — Grootendorst (2022), neural topic modelling
- [`Groq`](https://console.groq.com) — Llama 3.1 API used for CatLLM classification
