import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from utils import (
    set_seed,
    load_data,
    load_test_data
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Atention -------------------
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        attended = attn_weights @ V
        return attended.mean(dim=1)  

class AttentionFusionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.attn = SelfAttention(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_feats, txt_feats):
        x = torch.stack([img_feats, txt_feats], dim=1)  
        x = self.attn(x)  
        return self.fc(x)

def train_attention_classifier(model, train_loader, val_loader, optimizer, loss_fn, epochs=30, patience=5):
    best_loss = float('inf')
    best_model_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, txt, y in train_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            pred = model(img, txt)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, txt, y in val_loader:
                img, txt, y = img.to(device), txt.to(device), y.to(device)
                pred = model(img, txt)
                val_loss += loss_fn(pred, y).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[AttentionClassifier] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping attention classifier.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

class PairedDataset(Dataset):
    def __init__(self, img_feats, txt_feats, labels):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.img_feats[idx], self.txt_feats[idx], self.labels[idx]
    
# ---------------- Pipeline -------------------
def main(dataset, seed=42):
    print("Train dataset: " + dataset)

    set_seed(seed)

    data = load_data(f"train/{dataset}")
    train_art, _ = data["art"]["train"]
    val_art, _ = data["art"]["val"]
    train_mis, _ = data["misinformation"]["train"]
    val_mis, _ = data["misinformation"]["val"]
    train_sat, _ = data["satire"]["train"]
    val_sat, _ = data["satire"]["val"]

    image_dim = 768
    def split_feats(arr): return arr[:, :image_dim], arr[:, image_dim:]

    # Split features
    art_img, art_txt = split_feats(train_art)
    mis_img, mis_txt = split_feats(train_mis)
    sat_img, sat_txt = split_feats(train_sat)
    art_val_img, art_val_txt = split_feats(val_art)
    mis_val_img, mis_val_txt = split_feats(val_mis)
    sat_val_img, sat_val_txt = split_feats(val_sat)

    # Fuse features
    X_img = torch.tensor(np.vstack([art_img, mis_img, sat_img]), dtype=torch.float32)
    X_txt = torch.tensor(np.vstack([art_txt, mis_txt, sat_txt]), dtype=torch.float32)
    y_train = torch.tensor([0]*len(art_img) + [1]*len(mis_img) + [2]*len(sat_img), dtype=torch.long)

    X_val_img = torch.tensor(np.vstack([art_val_img, mis_val_img, sat_val_img]), dtype=torch.float32)
    X_val_txt = torch.tensor(np.vstack([art_val_txt, mis_val_txt, sat_val_txt]), dtype=torch.float32)
    y_val = torch.tensor([0]*len(art_val_img) + [1]*len(mis_val_img) + [2]*len(sat_val_img), dtype=torch.long)

    # Define model
    model = AttentionFusionClassifier(input_dim=image_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_attention_classifier(model,
        DataLoader(PairedDataset(X_img, X_txt, y_train), batch_size=32, shuffle=True),
        DataLoader(PairedDataset(X_val_img, X_val_txt, y_val), batch_size=32),
        optimizer, loss_fn)

    # Load test data
    X_test, y_test = load_test_data()
    img_feats, txt_feats = split_feats(X_test.numpy())
    img_feats = torch.tensor(img_feats, dtype=torch.float32)
    txt_feats = torch.tensor(txt_feats, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(img_feats.to(device), txt_feats.to(device)).argmax(dim=1).cpu().numpy()

    # Accuracy
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per per class metrics
    class_names = ['art', 'misinformation', 'satire']
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist() 
    report = classification_report(y_test, y_pred, target_names=['art', 'misinformation', 'satire'], digits=4, output_dict=True)

    return acc, cm, report


# Main
dataset = "image_guided" # Options: image_guided, description_guided, or multimodally_guided
results = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": []
    }

for i in range(10):
    print("------------------------ Trial: " +str(i+1)+ " ------------------------")
    acc, cm, report = main(dataset, 42+i)
    results["accuracies"].append(acc)
    results["confusion_matrices"].append(cm)
    results["classification_reports"].append(report)

# Compute stats
accs = np.array(results["accuracies"])
results["accuracy_mean"] = float(np.mean(accs))
results["accuracy_std"] = float(np.std(accs))

print(f"Average Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

with open(f"results/{dataset}/multimodal_attention_results.json", "w") as f:
    json.dump(results, f, indent=2)