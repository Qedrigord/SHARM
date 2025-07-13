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
    Classifier,
    train_classifier,
    PairedDataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fuse_features(img_feats, txt_feats, fusion):
    if fusion == "image":
        return img_feats
    elif fusion == "text":
        return txt_feats
    elif fusion == "concat":
        return np.hstack([img_feats, txt_feats])
    elif fusion == "add":
        return img_feats + txt_feats
    elif fusion == "subtract":
        return img_feats - txt_feats
    elif fusion == "multiply":
        return img_feats * txt_feats
    elif fusion == "casm":
        concat = np.hstack([img_feats, txt_feats])
        add = img_feats + txt_feats
        subtract = img_feats - txt_feats
        multiply = img_feats * txt_feats
        return np.hstack([concat, add, subtract, multiply])
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion}")


# ---------------- Pipeline -------------------
def main(dataset, seed, fusion):
    print("Train dataset: " + dataset)
    print("Fusion method: " + fusion)

    set_seed(seed)

    data = load_data(f"{dataset}")
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
    classifier = Classifier(input_dim=input_dim).to(device)
    train_loader = DataLoader(PairedDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(PairedDataset(X_val, y_val), batch_size=32)

    train_classifier(
        classifier,
        train_loader,
        val_loader,
        optimizer=AdamW(classifier.parameters(), lr=5e-5, weight_decay=1e-4),
        loss_fn=nn.CrossEntropyLoss()
    )

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

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON
    report = classification_report(y_test, y_pred, target_names=['art', 'misinformation', 'satire'], digits=4, output_dict=True)

    return acc, cm, report, 


# Main
dataset = "image_guided" # Options: image_guided, description_guided, or multimodally_guided
fusion_modes = ["image", "text", "concat", "add", "subtract", "multiply", "casm"] # Configure as desired

for fusion_mode in fusion_modes:
    print(f"\n--- Fusion Mode: {fusion_mode} ---")
    fusion_results = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": []
    }

    for i in range(10):
        print("------------------------ Trial: " +str(i+1)+ " ------------------------")
        acc, cm, report = main(dataset, 42+i, fusion_mode)
        
        fusion_results["accuracies"].append(acc)
        fusion_results["confusion_matrices"].append(cm)
        fusion_results["classification_reports"].append(report)

    # Compute stats
    accs = np.array(fusion_results["accuracies"])
    fusion_results["accuracy_mean"] = float(np.mean(accs))
    fusion_results["accuracy_std"] = float(np.std(accs))

    print(f"Average Accuracy ({fusion_mode}): {fusion_results['accuracy_mean']:.4f} ± {fusion_results['accuracy_std']:.4f}")

    # Save each fusion result to its own JSON file
    with open(f"results/{dataset}/{fusion_mode}_results.json", "w") as f:
        json.dump(fusion_results, f, indent=2)

print("\n Each fusion mode result saved separately.")