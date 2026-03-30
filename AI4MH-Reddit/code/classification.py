"""
Training mental/mental-roberta-base on train.csv to predict the classification for all posts/comments in test.csv
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

warnings.filterwarnings("ignore")

# CONSTANTS

RAW2INT = {-1:0, 0:1, 1:2, 99:3}
INT2RAW = {v:k for k,v in RAW2INT.items()}
LABEL_NAMES = ["negative","neutral","positive","not_mentioned"]
NUM_LABELS = len(RAW2INT)

# HYPERPARAMETERS

MODEL_NAME = "mental/mental-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
VALIDATION_SPLIT = 0.2
CONFIDENCE_THRESHOLD = 0.40
SEED = 42

# CLASS FOR LOADING DATASET

class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer 
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
# Utility Functions

def compute_class_weights(labels: list, device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    n_total = len(labels)
    weights = torch.zeros(NUM_LABELS, dtype=torch.float, device=device)
    for i in range(NUM_LABELS):
        weights[i]= n_total/(NUM_LABELS * counts.get(i,1))
    return weights

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion_matrix: Validation Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" Confusion Matrix save to {save_path}")

# Training/ Evaluation / Prediction Loops

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = correct = total = 0 
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ids, attention_mask=mask).logits
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == lbls).sum().item()
        total += len(lbls)

    return total_loss/len(loader), correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            logits = model(ids, attention_mask=mask).logits
            total_loss += criterion(logits, lbls).item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(lbls.cpu().numpy())
    return total_loss/len(loader), all_preds, all_true

def predict(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            probs = torch.softmax(model(ids, attention_mask=mask).logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    all_probs = np.array(all_probs)
    return all_probs.argmax(axis=1).astype(int), all_probs

# Combined Training Pipeline

def run(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n Loading Training Data ")
    train_df = pd.read_csv(args.train)
    print(f" Shape of Training Data: {train_df.shape}")

    print("\n Loading Test Data ")
    test_df = pd.read_csv(args.test)
    print(f" Shape of Test Data: {test_df.shape}")

    assert "cleaned_text" in train_df.columns, \
        "train.csv must have a 'cleaned_text' column. Run run_preprocessing.py first."
    assert "cleaned_text" in test_df.columns, \
        "test.csv must have a 'cleaned_text' column. Run run_preprocessing.py first."
    
    mask = (
        ~train_df["is_removed"].fillna(False).astype(bool)&
        ~train_df["is_image"].fillna(False).astype(bool)&
        ~train_df["too_short"].fillna(False).astype(bool)&
        train_df["functional"].notna()&
        train_df["cleaned_text"].notna()
    )
    train_df = train_df[mask].reset_index(drop=True)
    print(f"Shape of Training Data after Filtering: {train_df.shape}")

    # LABEL MAPPING FOR ROBERTA TO CLASSIFY
    train_df["label_int"] = train_df["functional"].astype(int).map(RAW2INT)
    unmapped = train_df["label_int"].isna().sum()
    if unmapped:
        print(" Dropping unscrupulous labels from training dataset")
        train_df = train_df.dropna(subset=["label_int"]).reset_index(drop=True)
    train_df["label_int"] = train_df["label_int"].astype(int)

    # Counting Labels 
    for raw, name in zip([-1,0,1,99],LABEL_NAMES):
        n = (train_df["functional"].astype(int) == raw).sum()
        print(f" {name:>14} ({raw:>3}): {n}")

    x_train, x_val, y_train, y_val = train_test_split(
        train_df["cleaned_text"].tolist(),
        train_df["label_int"].tolist(),
        test_size = VALIDATION_SPLIT,
        stratify = train_df["label_int"].tolist(),
        random_state = SEED,
    )

    print(f" SUMMARY: \n" 
          f"Training Dataset: {len(x_train)}"
          f"Testing Dataset: {len(x_val)}")
    
    print(f"\n Loading tokenizer ({MODEL_NAME})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.hf_token)

    num_workers = 0 if device.type=="cpu" else 2

    train_loader = DataLoader(
        RedditDataset(x_train,
                      y_train,
                      tokenizer, 
                      MAX_LENGTH,),
        batch_size = BATCH_SIZE,
        shuffle = True, 
        num_workers=num_workers,)
    
    val_loader = DataLoader(
        RedditDataset(x_val,
                      y_val,
                      tokenizer, 
                      MAX_LENGTH,),
        batch_size = BATCH_SIZE,
        shuffle = False, 
        num_workers=num_workers,)
    
    test_loader = DataLoader(
        RedditDataset(test_df["cleaned_text"].fillna("").tolist(),
                      None,
                      tokenizer, 
                      MAX_LENGTH,),
        batch_size = BATCH_SIZE,
        shuffle = False, 
        num_workers=num_workers,)
    
    print(f"Loading Model: ({MODEL_NAME})")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
        token = args.hf_token,
    ).to(device)

    class_weights = compute_class_weights(y_train, device)
    print(f"Class weights: { {LABEL_NAMES[i]: f'{w:.2f}' for i, w in enumerate(class_weights.cpu().numpy())} }")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = LEARNING_RATE,
                                  weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f" Training for {args.epochs} epochs")
    
    best_val_loss = float("inf")
    best_val_preds = None
    best_val_true = None
    best_model_dir = os.path.join(args.output_dir, "best_model")

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        val_acc = np.mean(np.array(val_preds) == np.array(val_true))

        print(f" Epoch {epoch}/{args.epochs} "
              f"train_loss = {tr_loss:.4f} | train_accuracy = {tr_acc:.4f} " 
              f"val_loss = {val_loss:.4f} | val_accuracy = {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_preds = val_preds
            best_val_true = val_true
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f" Best model so far saved to {best_model_dir}" 
                  f" best validation loss = {best_val_loss:.4f}")
            
    print(f"CLASSIFICATION REPORT (Validation Set Only)")
    report = classification_report(
        [LABEL_NAMES[i] for i in best_val_true],
        [LABEL_NAMES[i] for i in best_val_preds],
        zero_division=0,
    )
    print(report)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report\n{'-'*50}\n{report}")
    print(f"Classification Report saved to {report_path}")

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(best_val_true, best_val_preds, cm_path)

    print("Loading best model for test set classifications \n")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)

    test_preds, test_probs = predict(model, test_loader, device)
    test_df["functional_pred"] = [INT2RAW[i] for i in test_preds]
    test_df["functional_label"] = test_df["functional_pred"].map({
        -1:"negative",
        0:"neutral",
        1:"positive",
        99:"not_mentioned"
    })

    test_df["confidence"] = test_probs.max(axis=1).round(4)
    test_df["low_confidence"] = (test_df["confidence"] < CONFIDENCE_THRESHOLD)

    print(f"\n Predicted Label Distribution (Test Set): ")
    print(test_df["functional_label"].value_counts().to_string())
    print(f"\n  Low-confidence rows: {test_df['low_confidence'].sum()}")

    os.makedirs(args.data_dir, exist_ok=True)

    prediction_path = os.path.join(args.data_dir, "test_withPred.csv")
    test_df.to_csv(prediction_path, index=False)
    print(f" Save test Set Prediction here: {prediction_path}")

    low_conf_data_path = os.path.join(args.data_dir, "low_conf_review.csv")
    test_df[test_df["low_confidence"]].to_csv(low_conf_data_path, index=False)
    print(f" Save Low_Confidence Set Prediction here: {low_conf_data_path}")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MentalRoBERTa on Reddit Mental Health Data")
    parser.add_argument("--train",      default="train.csv",       help="Path to labelled training CSV")
    parser.add_argument("--test",       default="test.csv",        help="Path to test CSV")  
    parser.add_argument("--output_dir", default="model_output")
    parser.add_argument("--data_dir",   default="processed_input")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--hf_token",   default="")
    args = parser.parse_args()
    run(args)  


   
    







