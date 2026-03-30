'''
Training mental-health RoBERTa classification model on 800 labeled posts from train.csv and predicting on 
20,000 posts from test.csv.

We use the MentalRoBERTa model, initialized with RoBERTa-Base (cased_L-12_H-768_A-12) and trained with 
mental health-related posts collected from Reddit.
For more: https://huggingface.co/mental/mental-roberta-base
'''

import os 
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# try:
#     import wandb
#     wandb_available = True
# except ImportError:
#     wandb_available = False

warnings.filterwarnings("ignore")

# Model config

LABEL_COLS = ["functional", "relational", "metaphysical", "technical"]

INT_TO_LABEL = {0: -1, 1: 0, 2: 1, 3: 99}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}
LABEL_NAMES  = ["negative", "neutral", "positive", "not_mentioned"]

HF_TOKEN             = "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"
model_name           = "mental/mental-roberta-base"
max_length           = 256
batch_size           = 16
num_epochs           = 5
alpha                = 2e-5
warmup_ratio         = 0.1
validation_split     = 0.2
confidence_threshold = 0.40
output_dir           = "model_output"
seed                 = 42


class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length  = self.max_length,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt"
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def weighted_class(labels, device):
    from collections import Counter
    label_counts    = Counter(labels)
    n_total         = len(labels)
    n_total_classes = len(INT_TO_LABEL)
    class_weights   = torch.zeros(n_total_classes, dtype=torch.float).to(device)
    for i in range(n_total_classes):
        weight = n_total / (n_total_classes * label_counts.get(i, 1))
        class_weights[i] = weight
    return class_weights


def train_model(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        prediction  = outputs.logits.argmax(dim=1)
        correct    += (prediction == labels).sum().item()
        total      += len(labels)

    return total_loss / len(loader), correct / total


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs    = model(input_ids, attention_mask=attention_mask)
            loss       = criterion(outputs.logits, labels)
            total_loss += loss.item()

            all_predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), all_predictions, all_labels


def predict_labels(model, loader, device):
    model.eval()
    all_predictions, all_probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids, attention_mask=attention_mask)
            probabilities  = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probabilities.extend(probabilities)
            all_predictions.extend(probabilities.argmax(axis=1))

    return np.array(all_predictions, dtype=int), np.array(all_probabilities)


def plot_confusion_matrix(y_true, y_pred, factor, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"Confusion Matrix for {factor} Factor - Validation Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Single Factor Training

def train_singlefactor(factor: str,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       tokenizer,
                       device,
                       model_name: str,
                       use_wandb: bool = False):

    print(f"\n{'='*60}")
    print(f"  Factor: {factor}")
    print(f"{'='*60}")

    factor_dir = os.path.join(output_dir, factor)
    os.makedirs(factor_dir, exist_ok=True)

    # if use_wandb and wandb_available:
    #     wandb.init(...)
    factor_train = train_df.dropna(subset=[factor]).copy()
    factor_train["label_int"] = factor_train[factor].astype(int).map(LABEL_TO_INT)  
    factor_train = factor_train.dropna(subset=["label_int"])                        
    factor_train["label_int"] = factor_train["label_int"].astype(int)                

    print(f"  Training samples with valid '{factor}' label: {len(factor_train)}")
    print(f"  Label distribution:\n{factor_train[factor].value_counts().to_string()}")

    if len(factor_train) < 20:
        print("  Not enough rows to train model. Skipping.")
        return test_df

    # Train / val split 
    X_train, X_val, y_train, y_val = train_test_split(
        factor_train["cleaned_text"].tolist(),
        factor_train["label_int"].tolist(),
        test_size    = validation_split,
        stratify     = factor_train["label_int"].tolist(),
        random_state = seed,
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # DataLoaders 
    train_ds = RedditDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = RedditDataset(X_val,   y_val,   tokenizer, max_length)
    test_ds  = RedditDataset(
        test_df["cleaned_text"].fillna("").tolist(),
        None, tokenizer, max_length
    )

    num_workers = 0 if device.type == "cpu" else 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = len(INT_TO_LABEL),
        token      = HF_TOKEN,
    ).to(device)

    class_weights = weighted_class(y_train, device)
    print(f"  Class weights: { {LABEL_NAMES[i]: f'{w:.2f}' for i, w in enumerate(class_weights.cpu().numpy())} }")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=alpha, weight_decay=0.01)
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = warmup_steps,
        num_training_steps = total_steps,
    )

    # Training loop 
    best_val_loss  = float("inf")
    best_val_preds = None
    best_val_true  = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_preds, val_true = evaluate_model(model, val_loader, criterion, device)
        val_acc = np.mean(np.array(val_preds) == np.array(val_true))

        print(f"  Epoch {epoch}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        # if use_wandb and wandb_available:
        #     wandb.log({...})

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_preds = val_preds
            best_val_true  = val_true
            model.save_pretrained(os.path.join(factor_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(factor_dir, "best_model"))

    print(f"  ✓ Best model saved → {factor_dir}/best_model")

    # Evaluation report 
    report = classification_report(
        [INT_TO_LABEL[i] for i in best_val_true],
        [INT_TO_LABEL[i] for i in best_val_preds],
        zero_division=0,
    )
    print(f"\n  Classification Report ({factor}):\n{report}")

    report_path = os.path.join(factor_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {factor}\n{'='*50}\n{report}")

    cm_path = os.path.join(factor_dir, "confusion_matrix.png")
    plot_confusion_matrix(best_val_true, best_val_preds, factor, cm_path)

    # if use_wandb and wandb_available:
    #     wandb.log({...})
    #     wandb.finish()

    # Predict on test set 
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(factor_dir, "best_model")
    ).to(device)

    test_preds, test_probs = predict_labels(model, test_loader, device)

    test_df[f"{factor}_pred"]         = [INT_TO_LABEL[p] for p in test_preds]   
    test_df[f"{factor}_confidence"]   = test_probs.max(axis=1).round(4)
    test_df[f"{factor}_prob_neg"]     = test_probs[:, 0].round(4)
    test_df[f"{factor}_prob_neutral"] = test_probs[:, 1].round(4)
    test_df[f"{factor}_prob_pos"]     = test_probs[:, 2].round(4)
    test_df[f"{factor}_prob_99"]      = test_probs[:, 3].round(4)

    print(f"\n  Predicted distribution ({factor}):")
    dist = test_df[f"{factor}_pred"].value_counts()            
    print(dist.to_string())

    return test_df

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/train.csv")
    parser.add_argument("--test",  default="/Users/khnsakhnm/Documents/MentalHealth_Research/AI4MH-Reddit/data/processed_input/test.csv")
    parser.add_argument("--model", default=model_name,
                        help="HuggingFace model name (e.g. mental/mental-roberta-base)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    arg, _ = parser.parse_known_args()

    # use_wandb = wandb_available and not arg.no_wandb

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("⚠️  No GPU detected. Training will be slow. Use Google Colab (GPU runtime).")

    # Load data
    print(f"\nLoading {arg.train}...")
    train_df = pd.read_csv(arg.train)
    print(f"  Shape: {train_df.shape}")

    print(f"Loading {arg.test}...")
    test_df = pd.read_csv(arg.test)
    print(f"  Shape: {test_df.shape}")

    # Filter bad training rows
    mask = (
        ~train_df["is_removed"].fillna(False) &
        ~train_df["is_image"].fillna(False) &
        ~train_df["too_short"].fillna(False)
    )
    train_df = train_df[mask].reset_index(drop=True)
    print(f"Training samples after filtering: {len(train_df)}")

    # Load tokenizer
    print(f"\nLoading tokenizer ({arg.model})...")
    tokenizer = AutoTokenizer.from_pretrained(arg.model, token=HF_TOKEN)

    # Train one model per factor
    for factor in LABEL_COLS:
        test_df = train_singlefactor(
            factor, train_df, test_df, tokenizer, device, arg.model, use_wandb=False
        )

    # Flag low-confidence rows
    conf_cols = [f"{f}_confidence" for f in LABEL_COLS if f"{f}_confidence" in test_df.columns]
    if conf_cols:
        test_df["low_conf"] = (test_df[conf_cols] < confidence_threshold).any(axis=1)
    else:
        test_df["low_conf"] = False

    # Save predictions
    pred_path = "test_with_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"\n Saved => {pred_path}")

    # Save low-confidence rows for human review
    low_conf_df   = test_df[test_df["low_conf"]]         
    low_conf_path = "low_confidence_review.csv"
    low_conf_df.to_csv(low_conf_path, index=False)
    print(f"\n Saved => {low_conf_path}  "
          f"({len(low_conf_df)} rows, {len(low_conf_df)/len(test_df):.1%} of test set)")

    print("\n Classification complete.")
    print(f"   Predictions  → {pred_path}")
    print(f"   Review queue → {low_conf_path}")
    print(f"   Models saved → {output_dir}/<factor>/best_model/")


if __name__ == "__main__":   
    main()