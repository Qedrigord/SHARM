import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(folder):
    categories = ["art", "misinformation", "satire"]
    data = {}
    for cat in categories:
        x_train_1 = np.load(f"train data/{folder}/{cat}_train_part1.npy")
        x_train_2 = np.load(f"train data/{folder}/{cat}_train_part2.npy")
        x_train = np.concatenate([x_train_1, x_train_2])
        x_val = np.load(f"train data/{folder}/{cat}_val.npy")
        x_base_train = np.load(f"train data/{folder}/{cat}_base_train.npy")
        x_base_val = np.load(f"train data/{folder}/{cat}_base_val.npy")
        data[cat] = {
            "train": (x_train, x_base_train),
            "val": (x_val, x_base_val)
        }
    return data

def load_test_data():
    def load(name):
        return (np.load(f"test data/{name}_image_features.npy"),
                np.load(f"test data/{name}_text_features.npy"))

    art_img, art_txt = load("art")
    art_tw_img, art_tw_txt = load("art_tweets")
    mis_img, mis_txt = load("misinformation")
    sat_img, sat_txt = load("satire")
    sat_tw_img, sat_tw_txt = load("satire_tweets")

    art_feats = np.hstack([np.vstack([art_img, art_tw_img]), np.vstack([art_txt, art_tw_txt])])
    mis_feats = np.hstack([mis_img, mis_txt])
    sat_feats = np.hstack([np.vstack([sat_img, sat_tw_img]), np.vstack([sat_txt, sat_tw_txt])])
    
    X = np.vstack([art_feats, mis_feats, sat_feats])
    y = np.concatenate([
        np.zeros(art_feats.shape[0]),
        np.ones(mis_feats.shape[0]),
        np.full(sat_feats.shape[0], 2)
    ])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ----------------- Classifier -------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.GELU(), 
            nn.Linear(512, 128), 
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(model, train_loader, val_loader, optimizer, loss_fn, epochs=30, patience=3):
    best_loss = float('inf')
    best_model_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Classifier] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping classifier training.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


# ----------------- Utility -------------------
class PairedDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]