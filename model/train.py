import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import create_dataloaders  # your dataset.py
import os
from pathlib import Path
from tqdm import tqdm
from models import SimpleCNN, RandomForest, ViT

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
    random_crop_train = True
    epochs = 20
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints"
    checkpoint_interval = 5


# ------------------------------
# Training function
# ------------------------------
def train():
    cfg = Config()
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # Data
    dataloaders = create_dataloaders(
        zarr_dir=cfg.zarr_dir,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        random_crop_train=cfg.random_crop_train
    )

    # Model, loss, optimizer
    model = SimpleCNN(in_channels=1, out_channels=1).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        model.train()
        running_loss = 0.0

        for batch_idx, (patches, metadata) in enumerate(tqdm(dataloaders['train'])):
            patches = patches.to(cfg.device)  # (B, C, H, W)
            
            # In this example, let's do an autoencoder-style reconstruction
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, patches)  # reconstruct input
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloaders['train'])
        print(f"Train Loss: {avg_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, metadata in dataloaders['val']:
                patches = patches.to(cfg.device)
                outputs = model(patches)
                loss = criterion(outputs, patches)
                val_loss += loss.item()
        val_loss /= len(dataloaders['val'])
        print(f"Validation Loss: {val_loss:.6f}")

        # Save checkpoint
        if epoch % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    train()