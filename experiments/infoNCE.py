import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json

from utils import (
    set_seed,
    load_data,
    load_test_data,
    Classifier
)

# ---------------- CONFIG ---------------- #
TEMPERATURE = 0.07
PROJ_DIM = 128
CLASSIFIER_EPOCHS = 200
ALIGN_EPOCHS = 200
NUM_CLASSES = 3
BATCH_SIZE = 64
IMAGE_DIM = 768
PATIENCE = 25

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- MODELS ------------------ #
class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

# -------------- LOSS -------------------- #
def info_nce_loss(image_embeddings, text_embeddings, temperature=TEMPERATURE):
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    logits = image_embeddings @ text_embeddings.T
    logits = logits / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# ------------- LOAD DATA --------------- #
def train_infonce(train_loader, val_loader, input_dim):
    proj_image = ProjectionMLP(input_dim).to(device)
    proj_text = ProjectionMLP(input_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(proj_image.parameters()) + list(proj_text.parameters()),
        lr=1e-5,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(ALIGN_EPOCHS):
        proj_image.train()
        proj_text.train()
        total_train_loss = 0.0

        for batch_images, batch_texts in train_loader:
            im_proj = proj_image(batch_images)
            tx_proj = proj_text(batch_texts)
            loss = info_nce_loss(im_proj, tx_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        proj_image.eval()
        proj_text.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_images_batch, val_texts_batch in val_loader:
                val_img_proj = proj_image(val_images_batch)
                val_txt_proj = proj_text(val_texts_batch)
                val_loss = info_nce_loss(val_img_proj, val_txt_proj)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train InfoNCE: {avg_train_loss:.4f} | Val InfoNCE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                "proj_image": proj_image.state_dict(),
                "proj_text": proj_text.state_dict()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered during alignment at epoch {epoch+1}")
                break

    if best_state is not None:
        proj_image.load_state_dict(best_state["proj_image"])
        proj_text.load_state_dict(best_state["proj_text"])

    return proj_image.eval(), proj_text.eval()

def train_classifier(train_feats, train_labels, val_feats, val_labels):
    classifier = Classifier(input_dim=train_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(CLASSIFIER_EPOCHS):
        classifier.train()
        logits = classifier(train_feats)
        loss = criterion(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_feats)
            val_loss = criterion(val_logits, val_labels)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)

    return classifier.eval()

def main(dataset, seed):
    print("Train dataset: " + dataset)

    set_seed(seed)

    data = load_data(f"{dataset}")
    train_art, _ = data["art"]["train"]
    val_art, _ = data["art"]["val"]
    train_mis, _ = data["misinformation"]["train"]
    val_mis, _ = data["misinformation"]["val"]
    train_sat, _ = data["satire"]["train"]
    val_sat, _ = data["satire"]["val"]

    image_dim = 768
    train_art_img, train_art_txt = train_art[:, :image_dim], train_art[:, image_dim:]
    train_mis_img, train_mis_txt = train_mis[:, :image_dim], train_mis[:, image_dim:]
    train_sat_img, train_sat_txt = train_sat[:, :image_dim], train_sat[:, image_dim:]

    X_train = torch.tensor(np.vstack([train_art_img, train_mis_img, train_sat_img]), dtype=torch.float32)
    y_train = torch.tensor([0]*len(train_art_img) + [1]*len(train_mis_img) + [2]*len(train_sat_img))
    text_inputs = torch.tensor(np.vstack([train_art_txt, train_mis_txt, train_sat_txt]), dtype=torch.float32)

    X_val = torch.tensor(np.vstack([val_art[:, :768], val_mis[:, :768], val_sat[:, :768]]), dtype=torch.float32)
    y_val = torch.tensor([0]*len(val_art) + [1]*len(val_mis) + [2]*len(val_sat))
    val_text = torch.tensor(np.vstack([val_art[:, 768:], val_mis[:, 768:], val_sat[:, 768:]]), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, text_inputs), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, val_text), batch_size=BATCH_SIZE)

    proj_image, proj_text = train_infonce(train_loader, val_loader, X_train.shape[1])

    with torch.no_grad():
        train_feats = proj_image(X_train) + proj_text(text_inputs)
        val_feats = proj_image(X_val) + proj_text(val_text)

    classifier = train_classifier(train_feats, y_train, val_feats, y_val)
    
    X_test, y_test = load_test_data()
    test_images = X_test[:, :IMAGE_DIM].to(device)
    test_texts = X_test[:, IMAGE_DIM:].to(device)
    test_labels = y_test.to(device)

    with torch.no_grad():
        test_combined = proj_image(test_images) + proj_text(test_texts)
        test_preds = classifier(test_combined).argmax(dim=1)
        test_acc = (test_preds == test_labels).float().mean().item()

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    class_names = ['art', 'misinformation', 'satire']
    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(test_labels.cpu(), test_preds.cpu())
    report = classification_report(test_labels.cpu(), test_preds.cpu(), target_names=['art', 'misinformation', 'satire'], digits=4, output_dict=True)

    return test_acc, cm, report


# Main
dataset = "image_guided"    # Options: image_guided, description_guided, or multimodally_guided
results = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": []
    }
for i in range(10):
    print("------------------------ Trial: " +str(i+1)+ " ------------------------")
    acc, cm, report = main(dataset, 42+i)
    results["accuracies"].append(acc)
    results["confusion_matrices"].append(cm.tolist())
    results["classification_reports"].append(report)

# Compute stats
accs = np.array(results["accuracies"])
results["accuracy_mean"] = float(np.mean(accs))
results["accuracy_std"] = float(np.std(accs))
print(f"Average Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

# Save each fusion result to its own JSON file
with open(f"results/{dataset}/infoNCE_add_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Each fusion mode result saved separately.")