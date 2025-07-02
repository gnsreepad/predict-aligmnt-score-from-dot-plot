import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# CONFIGURATION
# -----------------------
TEST_JSON = "data_dotplots/metadata/test.json"
MODEL_PATH = "model/custom_cnn_regressor.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        img = Image.open(item["dotplot"]).convert("L")
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
# EVALUATION
# -----------------------
def evaluate_model():
    # Load test set
    test_loader = DataLoader(DotPlotDataset(TEST_JSON), batch_size=BATCH_SIZE)

    # Load trained model
    model = EnhancedCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Convert to numpy
    y_true = np.array(all_targets).flatten()
    y_pred = np.array(all_preds).flatten()

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n📊 Test Performance:")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R^2:   {r2:.4f}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Alignment Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs True Alignment Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model/test_custom_cnn_scatter_eval.png")
    plt.show()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    evaluate_model()