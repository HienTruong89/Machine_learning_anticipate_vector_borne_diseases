import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import transforms
from transformers import ViTModel

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import umap.umap_ as umap

# =====================================================
# CONFIG
# =====================================================
IMAGE_FOLDER = "C:/Users/truon/OneDrive/Desktop/Project by 2026/RGB_images_three_category_dataset/RGB"
LABELS_CSV = "C:/Users/truon/OneDrive/Desktop/Project by 2026/Labels_category_dataset/labels_3c.csv"
FEATURE_CACHE = "vit_features_3c.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# REPRODUCIBILITY
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =====================================================
# DATASET
# =====================================================
class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "path"]).convert("RGB")
        label = self.df.loc[idx, "label"]
        return self.transform(img), label

# =====================================================
# TRANSFORMS (ViT CORRECT)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# LOAD DATA
# =====================================================
def load_dataframe(img_dir, csv_path):
    df = pd.read_csv(csv_path, index_col="ID")
    df["path"] = df.index.map(lambda x: os.path.join(img_dir, f"{x}.png"))
    df = df[df["path"].apply(os.path.exists)]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["class_label"])
    return df, le

# =====================================================
# FEATURE EXTRACTION
# =====================================================
@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        cls = model(x).last_hidden_state[:, 0, :]
        feats.append(cls.cpu())
        labels.append(y)

    return torch.cat(feats), torch.cat(labels)

# =====================================================
# CLASSIFIER
# =====================================================
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# TRAIN / EVAL
# =====================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        preds.append(model(x).argmax(1).cpu())
        targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    return {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, average="weighted"),
        "recall": recall_score(targets, preds, average="weighted"),
        "f1": f1_score(targets, preds, average="weighted"),
        "cm": confusion_matrix(targets, preds),
        "preds": preds,
        "targets": targets
    }

# -----------------------------
# VISUALIZATION
# -----------------------------
def plot_umap(features, labels, class_names, n_samples=5000):
    idx = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
    feature_norm=StandardScaler().fit_transform(features[idx])
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1,metric="euclidean",spread=1.5, random_state=SEED)
    emb = reducer.fit_transform(feature_norm)
    
    label_names = [class_names[l] for l in labels[idx]]
    label_cat = pd.Categorical(
       label_names,
       categories=list(class_names),
       ordered=True
   )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=emb[:, 0], y=emb[:, 1],
        hue=label_cat,
        palette="tab10", s=20,alpha=0.85
    )
    plt.title("UMAP of ViT Features")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# =====================================================
# MAIN
# =====================================================
def main():
    df, le = load_dataframe(IMAGE_FOLDER, LABELS_CSV)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        ImageDataset(train_df, transform),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)

    test_loader = DataLoader(
        ImageDataset(test_df, transform),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    if os.path.exists(FEATURE_CACHE):
        data = torch.load(FEATURE_CACHE)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE)
        vit.requires_grad_(False)

        X_train, y_train = extract_features(vit, train_loader)
        X_test, y_test = extract_features(vit, test_loader)

        torch.save({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}, FEATURE_CACHE)

    clf = MLP(X_train.shape[1], len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=LR)

    train_feat_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True)

    test_feat_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE)

    for e in range(EPOCHS):
        loss = train_epoch(clf, train_feat_loader, criterion, optimizer)
        print(f"Epoch {e+1:03d} | Loss {loss:.4f}")

    results = evaluate(clf, test_feat_loader)
    print(results)

    plot_confusion_matrix(results["cm"],class_names=le.classes_)
    plot_umap(features=X_test.numpy(),labels=y_test.numpy(),class_names=le.classes_,n_samples=5000)

if __name__ == "__main__":
    main()


