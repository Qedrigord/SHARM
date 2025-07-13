import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from losses import SupConLoss

from utils import (
    set_seed,
    load_data,
    load_test_data,
    Classifier,
    train_classifier,
    PairedDataset
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Models -------------------
class ProjectionNet(nn.Module):
    def __init__(self, input_dim=768, embed_dim=768):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.GELU(), 
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.projector(x)

# ----------------- Triplet Loss -------------------
def create_contrastive_samples_from_self(X, y):
    anchors, positives, negatives = [], [], []
    class_indices = {int(cls.item()): (y == cls).nonzero(as_tuple=True)[0] for cls in torch.unique(y)}

    for idx in range(len(X)):
        a_feat, a_label = X[idx], y[idx].item()

        # Positive: randomly sample a *different* sample with same label
        same_class_idxs = class_indices[a_label]
        pos_idx = np.random.choice(same_class_idxs[same_class_idxs != idx].cpu().numpy())
        pos_feat = X[pos_idx]

        # Negative: randomly sample from a different class
        neg_classes = [cls for cls in class_indices if cls != a_label]
        neg_cls = np.random.choice(neg_classes)
        neg_feat = X[np.random.choice(class_indices[neg_cls].cpu().numpy())]

        anchors.append(a_feat)
        positives.append(pos_feat)
        negatives.append(neg_feat)

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def triplet_loss(anchor, positive, negative, margin=1.0):
    return F.relu(F.pairwise_distance(anchor, positive) - F.pairwise_distance(anchor, negative) + margin).mean()

def train_contrastive_self(model, X, y, val_X, val_y, optimizer, margin=1.0, epochs=30, patience=3):
    best_loss = float('inf')
    best_model_state = None
    wait = 0
    model.to(device).train()

    for epoch in range(epochs):
        model.train()
        anchors, positives, negatives = create_contrastive_samples_from_self(X, y)
        loader = DataLoader(torch.utils.data.TensorDataset(anchors, positives, negatives), batch_size=64, shuffle=True)

        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ae, pe, ne = model(a), model(p), model(n)
            loss = triplet_loss(ae, pe, ne, margin)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(loader)

        with torch.no_grad():
            model.eval()
            a, p, n = create_contrastive_samples_from_self(val_X, val_y)
            ae, pe, ne = model(a.to(device)), model(p.to(device)), model(n.to(device))
            val_loss = triplet_loss(ae, pe, ne, margin).item()

        print(f"[Contrastive] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping contrastive training.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

# ----------------- Quadruplet Loss -------------------
def create_quadruplet_samples(X, y):
    anchors, positives, negatives1, negatives2 = [], [], [], []
    class_indices = {int(cls.item()): (y == cls).nonzero(as_tuple=True)[0] for cls in torch.unique(y)}

    for idx in range(len(X)):
        a_feat, a_label = X[idx], y[idx].item()

        # Positive
        same_class_idxs = class_indices[a_label]
        pos_idx = np.random.choice(same_class_idxs[same_class_idxs != idx].cpu().numpy())
        p_feat = X[pos_idx]

        # Negative1
        neg_classes = [cls for cls in class_indices if cls != a_label]
        n1_class = np.random.choice(neg_classes)
        n1_feat = X[np.random.choice(class_indices[n1_class].cpu().numpy())]

        # Negative2 (from different class than n1)
        other_neg_classes = [cls for cls in neg_classes if cls != n1_class]
        n2_class = np.random.choice(other_neg_classes)
        n2_feat = X[np.random.choice(class_indices[n2_class].cpu().numpy())]

        anchors.append(a_feat)
        positives.append(p_feat)
        negatives1.append(n1_feat)
        negatives2.append(n2_feat)

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives1), torch.stack(negatives2)

def quadruplet_loss(anchor, positive, negative1, negative2, margin1=1.0, margin2=0.5):
    # Euclidean distances
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist1 = F.pairwise_distance(anchor, negative1)
    neg_dist2 = F.pairwise_distance(positive, negative2)

    # Quadruplet loss
    loss = (pos_dist ** 2) + F.relu(margin1 - neg_dist1 ** 2) + F.relu(margin2 - neg_dist2 ** 2)
    return loss.mean()


def train_quadruplet(model, X, y, val_X, val_y, optimizer, margin1=1.0, margin2=0.5, epochs=30, patience=3):
    best_loss = float('inf')
    best_model_state = None
    wait = 0
    model.to(device).train()

    for epoch in range(epochs):
        model.train()
        a, p, n1, n2 = create_quadruplet_samples(X, y)
        loader = DataLoader(torch.utils.data.TensorDataset(a, p, n1, n2), batch_size=64, shuffle=True)

        total_loss = 0
        for a, p, n1, n2 in loader:
            a, p, n1, n2 = a.to(device), p.to(device), n1.to(device), n2.to(device)
            ae, pe, ne1, ne2 = model(a), model(p), model(n1), model(n2)
            loss = quadruplet_loss(ae, pe, ne1, ne2, margin1, margin2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(loader)

        with torch.no_grad():
            model.eval()
            a, p, n1, n2 = create_quadruplet_samples(val_X, val_y)
            ae, pe, ne1, ne2 = model(a.to(device)), model(p.to(device)), model(n1.to(device)), model(n2.to(device))
            val_loss = quadruplet_loss(ae, pe, ne1, ne2, margin1, margin2).item()

        print(f"[Quadruplet] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping quadruplet training.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


# ----------------- Supervised Contrastive Loss -------------------
def train_supcontrastive(model, train_feats, train_labels, optimizer, val_feats=None, val_labels=None, temperature=0.07, epochs=30, patience=3):
    model.to(device).train()
    criterion = SupConLoss(temperature=temperature)
    train_loader = DataLoader(torch.utils.data.TensorDataset(train_feats, train_labels), batch_size=64, shuffle=True)

    val_loader = None
    if val_feats is not None and val_labels is not None:
        val_loader = DataLoader(torch.utils.data.TensorDataset(val_feats, val_labels), batch_size=64)

    best_loss = float("inf")
    best_model_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            z1 = model(x)
            z2 = model(x)
            views = torch.stack([z1, z2], dim=1)
            loss = criterion(views, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    z1 = model(x)
                    z2 = model(x)
                    views = torch.stack([z1, z2], dim=1)
                    loss = criterion(views, y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"[SupCon] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = avg_train_loss
            print(f"[SupCon] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping SupCon training.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


# ---------------- Pipeline -------------------
def main(dataset, seed, contrastive_loss):
    print("Train dataset: " + dataset)
    print("Contrastive Learning Loss Function: " + contrastive_loss)

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

    input_dim = X_train.shape[1]
    proj_model = ProjectionNet(input_dim=input_dim)

    if contrastive_loss == "triplet":
        train_contrastive_self(proj_model, X_train, y_train, X_val, y_val,  AdamW(proj_model.parameters(), lr=5e-5, weight_decay=1e-4))
    elif contrastive_loss == "quadruplet":        
        train_quadruplet(proj_model, X_train, y_train, X_val, y_val, AdamW(proj_model.parameters(), lr=5e-5, weight_decay=1e-4))
    elif contrastive_loss == "supcon":
        train_supcontrastive(proj_model, X_train, y_train, optimizer=AdamW(proj_model.parameters(), lr=5e-5, weight_decay=1e-4), val_feats=X_val, val_labels=y_val)
    else:
        raise ValueError("Invalid Loss Function.")
    
    proj_model.eval()
    with torch.no_grad():
        train_proj = proj_model(X_train.to(device)).cpu()
        val_proj = proj_model(X_val.to(device)).cpu()

    final_clf_input = torch.cat([train_proj, text_inputs], dim=1)
    clf_labels = y_train
    
    val_final_input = torch.cat([val_proj, val_text], dim=1)
    val_labels = torch.tensor([0]*len(val_art) + [1]*len(val_mis) + [2]*len(val_sat))
    
    input_dim = final_clf_input.shape[1]
    clf_model = Classifier(input_dim=input_dim).to(device)

    train_classifier(clf_model,
        DataLoader(PairedDataset(final_clf_input, clf_labels), batch_size=32, shuffle=True),
        DataLoader(PairedDataset(val_final_input, val_labels), batch_size=32),
        optimizer=AdamW(clf_model.parameters(), lr=5e-5, weight_decay=1e-4),
        loss_fn=nn.CrossEntropyLoss())

    X_test, y_test = load_test_data()
    X_test_img, X_test_txt = X_test[:, :768], X_test[:, 768:]

    with torch.no_grad():
        X_test_proj = proj_model(X_test_img.to(device)).cpu()
        X_test_final = torch.cat([X_test_proj, X_test_txt], dim=1)
        y_pred = clf_model(X_test_final.to(device)).argmax(dim=1).cpu().numpy()

    # Accuracy
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Accuracy per class
    class_names = ['art', 'misinformation', 'satire']
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON
    report = classification_report(y_test, y_pred, target_names=['art', 'misinformation', 'satire'], digits=4, output_dict=True)

    return acc, cm, report


# Main
dataset = "image_guided"        # Options: image_guided, description_guided, or multimodally_guided
contrastive_loss = "triplet"    # Options: triplet, quadruplet, supcon

results = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": []
    }
for i in range(10):
    print("------------------------ Trial: " +str(i+1)+ " ------------------------")
    acc, cm, report = main(dataset, 42+i, contrastive_loss)
    results["accuracies"].append(acc)
    results["confusion_matrices"].append(cm)
    results["classification_reports"].append(report)

# Compute stats
accs = np.array(results["accuracies"])
results["accuracy_mean"] = float(np.mean(accs))
results["accuracy_std"] = float(np.std(accs))
print(f"Average Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

# Save each fusion result to its own JSON file
with open(f"results/{dataset}/contrastive_{contrastive_loss}_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Each fusion mode result saved separately.")