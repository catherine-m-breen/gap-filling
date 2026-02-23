import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from dataset import create_dataloaders
from models import AttentionUNet


# ------------------------------
# Config
# ------------------------------
class Config:
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    batch_size = 16
    patch_size = 256
    stride = 128
    num_workers = 4
    normalize = True
    epochs = 20
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints_lobo"
    rf_max_pixels = 200_000  # subsample for RF


# ------------------------------
# Utility: collect basins
# ------------------------------
def get_unique_basins(dataset):
    return sorted(list(set(dataset.metadata["basin"])))


# ------------------------------
# UNet Training
# ------------------------------
def train_unet(train_loader, val_loader, cfg, fold_name):
    model = AttentionUNet(in_channels=11, out_channels=1).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0

        for X, Y, meta in train_loader:
            X, Y = X.to(cfg.device), Y.to(cfg.device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y, meta in val_loader:
            X, Y = X.to(cfg.device), Y.to(cfg.device)
            pred = model(X)
            val_loss += criterion(pred, Y).item()

    val_loss /= len(val_loader)

    torch.save(model.state_dict(),
               os.path.join(cfg.save_dir, f"unet_{fold_name}.pth"))

    return val_loss


# ------------------------------
# Random Forest Training
# ------------------------------
def train_random_forest(train_loader, val_loader, cfg, fold_name):

    X_list, Y_list = [], []

    # Collect pixel samples
    for X, Y, meta in train_loader:
        B, C, H, W = X.shape

        X = X.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        Y = Y.reshape(-1).numpy()

        X_list.append(X)
        Y_list.append(Y)

    X_all = np.concatenate(X_list)
    Y_all = np.concatenate(Y_list)

    # Subsample
    if len(X_all) > cfg.rf_max_pixels:
        idx = np.random.choice(len(X_all), cfg.rf_max_pixels, replace=False)
        X_all = X_all[idx]
        Y_all = Y_all[idx]

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1
    )

    rf.fit(X_all, Y_all)

    # Validation
    val_preds, val_targets = [], []

    for X, Y, meta in val_loader:
        B, C, H, W = X.shape
        X_flat = X.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        preds = rf.predict(X_flat)
        val_preds.append(preds)
        val_targets.append(Y.reshape(-1).numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    val_mse = mean_squared_error(val_targets, val_preds)

    joblib.dump(rf,
                os.path.join(cfg.save_dir, f"rf_{fold_name}.joblib"))

    return val_mse


# ------------------------------
# LOBO Training Loop
# ------------------------------
def train_lobo():

    cfg = Config()
    Path(cfg.save_dir).mkdir(exist_ok=True)

    # Load full training dataset (exclude test basins manually here)
    dataloaders_full = create_dataloaders(...)

    dataset = dataloaders_full["train"].dataset
    basins = sorted(list(set(dataset.basins)))

    results = {}

    for val_basin in basins:

        print(f"\n===== LOBO: Holding out {val_basin} =====")

        train_loader, val_loader = create_dataloaders(
            holdout_basin=val_basin
        )

        fold_name = val_basin.replace(" ", "_")

        unet_val = train_unet(train_loader, val_loader, cfg, fold_name)
        rf_val = train_random_forest(train_loader, val_loader, cfg, fold_name)

        results[val_basin] = {
            "unet_val_loss": unet_val,
            "rf_val_mse": rf_val
        }

    print("\nLOBO Results:")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    train_lobo()