import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from utils import (
    set_seed,
    load_data,
    load_test_data,
    Classifier,
    train_classifier,
    PairedDataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Models -------------------
class MLPReconstructor(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768), 
            nn.GELU(),
            nn.Linear(768, 768), 
            nn.GELU(),
            nn.Linear(768, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------- Training Functions -------------------
def train_reconstructor(model, train_loader, val_loader, epochs=30, patience=3):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    best_loss = float('inf')
    best_model_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
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
                val_loss += F.mse_loss(pred, y).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Reconstructor] Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping reconstructor training.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

def get_reconstructed_inputs(proj_feats, reconstructor_models):
    return torch.cat([m(proj_feats.to(device)).cpu() for m in reconstructor_models], dim=1)

# ---------------- Pipeline -------------------
def main(dataset, seed, reconstruction_config, mode):
    print("Train dataset: " + dataset)
    print("Reconstructors configuration: " + reconstruction_config)
    print("Classification input: " + mode)

    set_seed(seed)

    data = load_data(f"{dataset}")
    train_art, art_base_train = data["art"]["train"]
    val_art, art_base_val = data["art"]["val"]
    train_mis, mis_base_train = data["misinformation"]["train"]
    val_mis, mis_base_val = data["misinformation"]["val"]
    train_sat, sat_base_train = data["satire"]["train"]
    val_sat, sat_base_val = data["satire"]["val"]

    image_dim = 768
    train_art_img, train_art_txt = train_art[:, :image_dim], train_art[:, image_dim:]
    train_mis_img, train_mis_txt = train_mis[:, :image_dim], train_mis[:, image_dim:]
    train_sat_img, train_sat_txt = train_sat[:, :image_dim], train_sat[:, image_dim:]

    # Concatenate training and validation data
    X_train = torch.tensor(np.vstack([train_art_img, train_mis_img, train_sat_img]), dtype=torch.float32)
    text_inputs = torch.tensor(np.vstack([train_art_txt, train_mis_txt, train_sat_txt]), dtype=torch.float32)
    y_train = torch.tensor([0]*len(train_art_img) + [1]*len(train_mis_img) + [2]*len(train_sat_img))
   
    X_val = torch.tensor(np.vstack([val_art[:, :768], val_mis[:, :768], val_sat[:, :768]]), dtype=torch.float32)
    val_text = torch.tensor(np.vstack([val_art[:, 768:], val_mis[:, 768:], val_sat[:, 768:]]), dtype=torch.float32)
    y_val = torch.tensor([0]*len(val_art) + [1]*len(val_mis) + [2]*len(val_sat))
    
    X_test, y_test = load_test_data()
    X_test_img, X_test_txt = X_test[:, :768], X_test[:, 768:]

    if reconstruction_config == "class_specific_reconstructors":
        r_art, r_mis, r_sat = MLPReconstructor(), MLPReconstructor(), MLPReconstructor()

        train_reconstructor(r_art,
            DataLoader(PairedDataset(train_art_img, torch.tensor(art_base_train, dtype=torch.float32)), batch_size=32, shuffle=True),
            DataLoader(PairedDataset(val_art[:, :768], torch.tensor(art_base_val, dtype=torch.float32)), batch_size=32))

        train_reconstructor(r_mis,
            DataLoader(PairedDataset(train_mis_img, torch.tensor(mis_base_train, dtype=torch.float32)), batch_size=32, shuffle=True),
            DataLoader(PairedDataset(val_mis[:, :768], torch.tensor(mis_base_val, dtype=torch.float32)), batch_size=32))

        train_reconstructor(r_sat,
            DataLoader(PairedDataset(train_sat_img, torch.tensor(sat_base_train, dtype=torch.float32)), batch_size=32, shuffle=True),
            DataLoader(PairedDataset(val_sat[:, :768], torch.tensor(sat_base_val, dtype=torch.float32)), batch_size=32))
    
        with torch.no_grad():
            recon_inputs = get_reconstructed_inputs(X_train, [r_art, r_mis, r_sat])
            val_recon = get_reconstructed_inputs(X_val, [r_art, r_mis, r_sat])
            test_recon = get_reconstructed_inputs(X_test_img, [r_art, r_mis, r_sat])

    elif reconstruction_config == "shared_reconstructor":
        base_train = torch.tensor(np.concatenate([art_base_train, mis_base_train, sat_base_train]), dtype=torch.float32)
        base_val = torch.tensor(np.concatenate([art_base_val, mis_base_val, sat_base_val]), dtype=torch.float32)

        recon = MLPReconstructor()

        train_reconstructor(recon,
            DataLoader(PairedDataset(X_train, base_train.float()), batch_size=32, shuffle=True),
            DataLoader(PairedDataset(X_val, base_val.float()), batch_size=32))
        
        with torch.no_grad():
            recon_inputs = get_reconstructed_inputs(X_train, [recon])
            val_recon = get_reconstructed_inputs(X_val, [recon])
            test_recon = get_reconstructed_inputs(X_test_img, [recon])
    else:
        raise ValueError("Wrong reconstructor config.")
    
    if mode == "replace":
        final_clf_input = torch.cat([recon_inputs, text_inputs], dim=1)
        val_final_input = torch.cat([val_recon, val_text], dim=1)
        X_test_final = torch.cat([test_recon, X_test_txt], dim=1)
    elif mode == "combine":
        final_clf_input = torch.cat([X_train, recon_inputs, text_inputs], dim=1)
        val_final_input = torch.cat([X_val, val_recon, val_text], dim=1)
        X_test_final = torch.cat([X_test_img, test_recon, X_test_txt], dim=1)
    else:
        raise ValueError("Wrong mode.")


    input_dim = final_clf_input.shape[1]
    clf_model = Classifier(input_dim=input_dim).to(device)
    train_classifier(clf_model,
        DataLoader(PairedDataset(final_clf_input, y_train), batch_size=32, shuffle=True),
        DataLoader(PairedDataset(val_final_input, y_val), batch_size=32),
        optimizer=AdamW(clf_model.parameters(), lr=5e-5, weight_decay=1e-4),
        loss_fn=nn.CrossEntropyLoss())

    with torch.no_grad():
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
dataset = "image_guided"                            # Options: image_guided, description_guided, or multimodally_guided
reconstruction_config = "class_specific_reconstructors"      # Options: shared_reconstructor, class_specific_reconstructors
mode = "combine"                                    # Options: replace, combine

results = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": []
    }
for i in range(10):
    print("------------------------ Trial: " +str(i+1)+ " ------------------------")
    acc, cm, report = main(dataset, 42+i, reconstruction_config, mode)
    results["accuracies"].append(acc)
    results["confusion_matrices"].append(cm)
    results["classification_reports"].append(report)

# Compute stats
accs = np.array(results["accuracies"])
results["accuracy_mean"] = float(np.mean(accs))
results["accuracy_std"] = float(np.std(accs))
print(f"Average Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

# Save each fusion result to its own JSON file
with open(f"results/{dataset}/{reconstruction_config}_{mode}_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Each fusion mode result saved separately.")