import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridDataset(Dataset):
    def __init__(self, image_folder_path, labels_folder_path, num_classes, grid_size):
        self.image_folder_path = image_folder_path
        self.labels_folder_path = labels_folder_path
        self.num_classes = num_classes
        self.grid_size = grid_size
        
        self.image_filenames = os.listdir(image_folder_path)
        self.label_filenames = [f"{os.path.splitext(filename)[0]}.txt" for filename in self.image_filenames]

    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.image_folder_path, self.image_filenames[index])
        image_np, size = self.load_image(image_path)

        # Load labels
        label_path = os.path.join(self.labels_folder_path, self.label_filenames[index])
        data = self.load_data(label_path, size)

        self.image_folder = self.generate_grid_data(image_np, data)
    
        return self.image_folder

    def __len__(self):
        return len(self.image_filenames)

    def load_image(self, image_path):
        image_np = cv2.imread(image_path)
        size = image_np.shape
        return image_np, size[:2]

    def load_data(self, label_path, size):
        data = []
        with open(label_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                class_index, x1, y1, x2, y2 = map(float, line.strip().split())
                w, h = size
                data.append({'class_index': int(class_index), 'x1': round(x1 * w), 'y1': round(y1 * h), 'x2': round(x2 * w), 'y2': round(y2 * h)})
        return data
    
    def generate_grid_data(self, image, data):
        grid_h, grid_w = self.grid_size
        img_h, img_w = image.shape[:2]

        cell_h = img_h // grid_h
        cell_w = img_w // grid_w

        grid_images = []

        for cell_y in range(grid_h):
            for cell_x in range(grid_w):
                cell_top = cell_y * cell_h
                cell_bottom = cell_top + cell_h
                cell_left = cell_x * cell_w
                cell_right = cell_left + cell_w
                
                grid_cell_image = image[cell_top:cell_bottom, cell_left:cell_right, :]
                
                for label in data:
                    x1, y1, x2, y2, class_index = (
                        label['x1'],
                        label['y1'],
                        label['x2'],
                        label['y2'],
                        label['class_index'],
                    )

                    if (
                        (cell_left <= x1 <= cell_right or cell_left <= x2 <= cell_right)
                        and (cell_top <= y1 <= cell_bottom or cell_top <= y2 <= cell_bottom)
                    ):
                        grid_images.append([grid_cell_image, class_index])
                    else:
                        grid_images.append([grid_cell_image, None])

        return grid_images
    
class ClassifierDataset(Dataset):
    pass

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # Separate RGB channels and normalize the map
        self.rgb_norm = nn.BatchNorm2d(3)

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Object detection layers
        self.object_detection = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Separate RGB channels and normalize the map
        x = self.rgb_norm(x)

        # Feature extraction
        features = self.features(x)

        # Object detection
        output = self.object_detection(features)

        return output

def train_model(model, dataset, batch_size, num_epochs, learning_rate):
    # Create DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the device
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _, grids in dataloader:
            # Move images to the device
            images = images.to(device)

            # Prepare labels
            labels = []
            for grid_class in grids.keys():
                if grid_class == 'none':
                    label = torch.tensor([0], dtype=torch.float32).to(device)  # Label 0 for "none" class
                else:
                    label = torch.tensor([1], dtype=torch.float32).to(device)  # Label 1 for other classes
                labels.append(label)

            # Forward pass
            outputs = model(images)

            # Flatten the outputs and labels for the loss computation
            outputs = torch.cat([output.flatten() for output in outputs], dim=1)
            labels = torch.cat(labels, dim=1)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print the average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

image_folder_path = "./store/images"
labels_folder_path = "./store/labels/txt"
grid_size = (2, 2)
num_classes = 2  # Two classes: "none" or other labels
batch_size = 16
num_epochs = 10
learning_rate = 0.001

# Create an instance of the dataset
dataset = GridDataset(image_folder_path, labels_folder_path, num_classes, grid_size)

for i in range(1):
    image_folder = dataset[i]
    for j in image_folder:
        cv2.imshow('e', j[0])
        cv2.waitKey(0)
    print(image_folder)
