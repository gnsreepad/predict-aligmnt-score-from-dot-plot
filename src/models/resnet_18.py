import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "data_dotplots/metadata"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
VAL_JSON = os.path.join(DATA_DIR, "val.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

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
            transforms.Grayscale(num_output_channels=3),  # Convert 1-channel → 3-channel
            transforms.Resize((224, 224)),  # ResNet default input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet pretrained normalization
                                 std=[0.229, 0.224, 0.225])
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
def get_resnet18_regressor():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Replace classifier with regression output
    return model

# -----------------------
# TRAINING LOOP
# -----------------------
def train_model():
    train_loader = DataLoader(DotPlotDataset(TRAIN_JSON), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DotPlotDataset(VAL_JSON), batch_size=BATCH_SIZE)
    test_loader = DataLoader(DotPlotDataset(TEST_JSON), batch_size=BATCH_SIZE)

    model = get_resnet18_regressor().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y)

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
                val_loss += loss_fn(pred, y).item() * x.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader.dataset):.4f} | "
              f"Val Loss: {val_loss/len(val_loader.dataset):.4f}")

    torch.save(model.state_dict(), "resnet18_dotplot_regressor.pth")
    print("Model saved as resnet18_dotplot_regressor.pth")

    test_model(model, test_loader, loss_fn)

# -----------------------
# TEST LOOP
# -----------------------
def test_model(model, test_loader, loss_fn):
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

    print(f"Final Test MSE: {test_loss / len(test_loader.dataset):.4f}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    train_model()