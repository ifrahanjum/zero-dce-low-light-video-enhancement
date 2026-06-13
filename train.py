import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from core.model import DCENet
from core.losses import compute_losses

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "zero_dce_best.pth"

class LowLightDataset(Dataset):
    def __init__(self, image_paths: list[str], size: int = IMAGE_SIZE):
        self.paths = image_paths
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))

def build_dataloaders(data_dir: str):
    all_paths = glob.glob(os.path.join(data_dir, "**/*.*"), recursive=True)
    all_paths = [p for p in all_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.seed(42)
    random.shuffle(all_paths)

    train_end = int(len(all_paths) * 0.8)
    val_end = int(len(all_paths) * 0.9)

    drop = True if len(all_paths) > 50 else False

    train_dl = DataLoader(LowLightDataset(all_paths[:train_end]), batch_size=BATCH_SIZE, shuffle=True, drop_last=drop)
    val_dl   = DataLoader(LowLightDataset(all_paths[train_end:val_end]), batch_size=BATCH_SIZE, shuffle=False, drop_last=drop)
    return train_dl, val_dl

def run_epoch(model, loader, optimizer=None, train=True):
    model.train(train)
    totals = {k: 0.0 for k in ["total_loss", "illumination_smoothness_loss", "spatial_constancy_loss", "color_constancy_loss", "exposure_loss"]}

    if len(loader) == 0:
        return totals

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            if train: optimizer.zero_grad()
            
            curve_maps = model(batch)
            losses = compute_losses(batch, curve_maps)
            
            if train:
                losses["total_loss"].backward()
                optimizer.step()
                
            for k in totals: totals[k] += losses[k].item()
            
    return {k: v / len(loader) for k, v in totals.items()}

def train(data_dir: str):
    train_dl, val_dl = build_dataloaders(data_dir)
    model = DCENet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_dl, optimizer, train=True)
        val_metrics   = run_epoch(model, val_dl, train=False)
        
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_metrics['total_loss']:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")
        
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            torch.save(model.state_dict(), WEIGHTS_PATH)

if __name__ == "__main__":
    # Change this path to wherever your images are saved locally
    train(r"C:\Users\aaqui\Desktop\test_dataset")