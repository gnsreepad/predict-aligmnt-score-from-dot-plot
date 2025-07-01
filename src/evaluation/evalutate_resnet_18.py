import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# CONFIGURATION
# -----------------------
TEST_JSON = "data_dotplots/metadata/test.json"
MODEL_PATH = "model/resnet18_dotplot_regressor_test_mse_6.18.pth"
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
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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
def get_resnet18_regressor():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# -----------------------
# EVALUATION
# -----------------------
def evaluate_model():
    # Load data
    test_loader = DataLoader(DotPlotDataset(TEST_JSON), batch_size=BATCH_SIZE)

    # Load model
    model = get_resnet18_regressor().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Evaluate
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Convert to arrays
    y_true = np.array(all_targets).flatten()
    y_pred = np.array(all_preds).flatten()

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Mean Squared Error (MSE):      {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2):                {r2:.4f}")

    # Plot predictions vs true values
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Alignment Score")
    plt.ylabel("Predicted Alignment Score")
    plt.title("Predicted vs True Alignment Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dotplot_regression_scatter.png")
    plt.show()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    evaluate_model()