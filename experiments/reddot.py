import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json 
from utils import (
    set_seed,
    load_data,
    load_test_data,
    train_classifier,
    PairedDataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fuse_features(img_feats, txt_feats, fusion):
    if fusion == "reddot":
        concat = np.concatenate([img_feats, txt_feats], axis=1)  
        concat = concat.reshape(-1, 2, 768)  # [B, 2, 768]

        add = img_feats + txt_feats
        subtract = img_feats - txt_feats
        multiply = img_feats * txt_feats
        ops = np.stack([add, subtract, multiply], axis=1)  # [B, 3, 768]

        return np.concatenate([concat, ops], axis=1)  # [B, 5, 768]
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion}")

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_layers=2, num_heads=4, num_classes=3):
        super().__init__()
        self.token_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), 
            nn.GELU(), 
            nn.Linear(512, 128), 
            nn.GELU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, 1, -1) 
        x = torch.cat([cls_token, x], dim=1)  
        x = self.transformer(x)
        cls_output = x[:, 0, :]  
        return self.classifier(cls_output)

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

        # Validation
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

# ---------------- Pipeline -------------------
def main(dataset, seed):
    print("Train dataset: " + dataset)

    set_seed(seed)

    data = load_data(f"train/{dataset}")
    train_art, _ = data["art"]["train"]
    val_art, _ = data["art"]["val"]
    train_mis, _ = data["misinformation"]["train"]
    val_mis, _ = data["misinformation"]["val"]
    train_sat, _ = data["satire"]["train"]
    val_sat, _ = data["satire"]["val"]

    # Split features
    image_dim = 768
    def split_feats(arr): return arr[:, :image_dim], arr[:, image_dim:]

    art_img, art_txt = split_feats(train_art)
    mis_img, mis_txt = split_feats(train_mis)
    sat_img, sat_txt = split_feats(train_sat)

    art_val_img, art_val_txt = split_feats(val_art)
    mis_val_img, mis_val_txt = split_feats(val_mis)
    sat_val_img, sat_val_txt = split_feats(val_sat)

    # Fuse features
    fusion = "reddot"
    fused_train = np.vstack([
        fuse_features(art_img, art_txt, fusion),
        fuse_features(mis_img, mis_txt, fusion),
        fuse_features(sat_img, sat_txt, fusion)
    ])
    X_train = torch.tensor(fused_train, dtype=torch.float32)
    y_train = torch.tensor([0]*len(art_img) + [1]*len(mis_img) + [2]*len(sat_img), dtype=torch.long)

    fused_val = np.vstack([
        fuse_features(art_val_img, art_val_txt, fusion),
        fuse_features(mis_val_img, mis_val_txt, fusion),
        fuse_features(sat_val_img, sat_val_txt, fusion)
    ])
    X_val = torch.tensor(fused_val, dtype=torch.float32)
    y_val = torch.tensor([0]*len(art_val_img) + [1]*len(mis_val_img) + [2]*len(sat_val_img), dtype=torch.long)

    # Define model
    input_dim = X_train.shape[1]
    classifier = TransformerClassifier(input_dim=input_dim).to(device)
    train_classifier(classifier,
        DataLoader(PairedDataset(X_train, y_train), batch_size=32, shuffle=True),
        DataLoader(PairedDataset(X_val, y_val), batch_size=32),
        optimizer=AdamW(classifier.parameters(), lr=5e-5, weight_decay=1e-4),
        loss_fn=nn.CrossEntropyLoss())

    # Load test data
    X_test, y_test = load_test_data()
    img_feats_test, txt_feats_test = split_feats(X_test.numpy())
    fused_test = fuse_features(img_feats_test, txt_feats_test, fusion)
    X_test_fused = torch.tensor(fused_test, dtype=torch.float32)

    classifier.eval()
    with torch.no_grad():
        y_pred = classifier(X_test_fused.to(device)).argmax(dim=1).cpu().numpy()

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

fusion_results = {
    "accuracies": [],
    "confusion_matrices": [],
    "classification_reports": []
}

for i in range(10):
    print("------------------------ Trial: " +str(i+1)+ " ------------------------")
    acc, cm, report = main(dataset, 42+i)
    fusion_results["accuracies"].append(acc)
    fusion_results["confusion_matrices"].append(cm)
    fusion_results["classification_reports"].append(report)

# Compute stats
accs = np.array(fusion_results["accuracies"])
fusion_results["accuracy_mean"] = float(np.mean(accs))
fusion_results["accuracy_std"] = float(np.std(accs))

print(f"Average Accuracy (reddot): {fusion_results['accuracy_mean']:.4f} ± {fusion_results['accuracy_std']:.4f}")


with open(f"results/{dataset}/reddot_results.json", "w") as f:
    json.dump(fusion_results, f, indent=2)
