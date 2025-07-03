import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "data_dotplots/metadata"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
VAL_JSON = os.path.join(DATA_DIR, "val.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EARLY_STOPPING_PATIENCE = 7

# -----------------------
# REPRODUCIBILITY
# -----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------
# DATASET
# -----------------------
class DotPlotDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["dotplot"])
        img = self.transform(img)
        score = torch.tensor(item["score"], dtype=torch.float32)
        return img, score

# -----------------------
# MODEL
# -----------------------
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# -----------------------
# TRAINING LOOP
# -----------------------
def train_model():
    train_loader = DataLoader(DotPlotDataset(TRAIN_JSON), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DotPlotDataset(VAL_JSON), batch_size=BATCH_SIZE)
    test_loader = DataLoader(DotPlotDataset(TEST_JSON), batch_size=BATCH_SIZE)

    model = EnhancedCNN().to(DEVICE)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            pred = model(x)
            loss = 0.7 * mse_loss(pred, y) + 0.3 * mae_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                pred = model(x)
                val_loss += mse_loss(pred, y).item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model/custom_cnn_regressor.pth")
            print("Model saved with improved validation loss")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model/loss_curve.png")
    plt.show()

    test_model(model, test_loader, mse_loss)
# -----------------------
# TEST LOOP
# -----------------------
def test_model(model, test_loader, loss_fn):
    model.load_state_dict(torch.load("model/custom_cnn_regressor.pth"))
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            pred = model(x)
            test_loss += loss_fn(pred, y).item() * x.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print(f"\U0001F4CA Final Test MSE: {test_loss:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6)
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.title('Predicted vs True Scores')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model/test_custom_cnn_predictions_scatter.png")
    plt.show()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    train_model()