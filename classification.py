import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SmallSeparableCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallSeparableCNN, self).__init__()
        self.features = nn.Sequential(
            SeparableConv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SeparableConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model
num_classes = 2
model = SmallSeparableCNN(num_classes)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the transform for the test dataset (same as the transform used during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_single_image(image_path, model, transform, device, grid_size):
    image_np = cv2.imread(image_path)

    # Get the image size and grid size
    img_h, img_w = image_np.shape[:2]
    grid_h, grid_w = grid_size

    # Calculate the cell size
    cell_h = img_h // grid_h
    cell_w = img_w // grid_w

    # Initialize the grid predictions
    grid_predictions = np.zeros((grid_h, grid_w), dtype=int)

    # Perform inference on each grid cell
    grid_cell_images = np.zeros((grid_h * grid_w, 3, 224, 224), dtype=np.float32)
    for cell_y in range(grid_h):
        for cell_x in range(grid_w):
            # Extract the grid cell from the image
            cell_top = cell_y * cell_h
            cell_bottom = cell_top + cell_h
            cell_left = cell_x * cell_w
            cell_right = cell_left + cell_w
            grid_cell_image = image_np[cell_top:cell_bottom, cell_left:cell_right]

            # Apply the transform to the grid cell image
            grid_cell_image = transform(grid_cell_image)
            grid_cell_images[cell_y * grid_w + cell_x] = grid_cell_image

    # Convert the NumPy array to a PyTorch tensor and move it to the appropriate device
    grid_cell_images = torch.tensor(grid_cell_images).to(device)

    # Perform inference on all grid cells in parallel
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        outputs = model(grid_cell_images)
        _, predicted = torch.max(outputs.data, 1)

    # Reshape the predictions back to the grid shape
    predicted = predicted.cpu().numpy()
    grid_predictions = predicted.reshape(grid_h, grid_w)

    return grid_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time

for i in range(170):
    start_time = time.time()
    test_image_path = f"./store/images/frame_{i}.jpg"
    grid_size = (7, 7)  # Set the desired grid size
    predicted_grid = classify_single_image(test_image_path, model, transform, device, grid_size)
    end_time = time.time()
    print(f"frame_{i}, time: {end_time - start_time}")