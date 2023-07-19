import os
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class GridDataset(Dataset):
    def __init__(self, image_folder_path, labels_folder_path, grid_size):
        self.image_folder_path = image_folder_path
        self.labels_folder_path = labels_folder_path
        self.grid_size = grid_size

        self.image_filenames = os.listdir(image_folder_path)
        self.label_filenames = [f"{os.path.splitext(filename)[0]}.txt" for filename in self.image_filenames]

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder_path, self.image_filenames[index])
        image_np, size = self.load_image(image_path)

        label_path = os.path.join(self.labels_folder_path, self.label_filenames[index])
        data = self.load_data(label_path, size)

        grid_data = self.generate_grid_data(image_np, data)

        return grid_data

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
                data.append({
                    'class_index': int(class_index),
                    'x1': round(x1 * w),
                    'y1': round(y1 * h),
                    'x2': round(x2 * w),
                    'y2': round(y2 * h)
                })
        return data

    def generate_grid_data(self, image, data):
        grid_h, grid_w = self.grid_size
        img_h, img_w = image.shape[:2]
        cell_h = img_h // grid_h
        cell_w = img_w // grid_w

        grid_images = [[] for _ in range(len(data) + 1)]

        for cell_y in range(grid_h):
            for cell_x in range(grid_w):
                cell_top = cell_y * cell_h
                cell_bottom = cell_top + cell_h
                cell_left = cell_x * cell_w
                cell_right = cell_left + cell_w
                
                grid_cell_image = image[cell_top:cell_bottom, cell_left:cell_right]
                for label in data:
                    x1, y1, x2, y2, class_index = (
                        label['x1'],
                        label['y1'],
                        label['x2'],
                        label['y2'],
                        label['class_index'],
                    )

                    if (any(cell_left <= x <= cell_right for x in [x1, x2]) and any(cell_top <= y <= cell_bottom for y in [y1, y2])) or \
                    (x1 <= cell_left <= cell_right <= x2 and y1 <= cell_top <= cell_bottom <= y2) or \
                    (any(cell_left <= x <= cell_right for x in [x1, x2]) and y1 <= cell_top <= cell_bottom <= y2) or \
                    (any(cell_top <= y <= cell_bottom for y in [y1, y2]) and x1 <= cell_left <= cell_right <= x2):

                        grid_images[class_index + 1].append(grid_cell_image)
                    else:
                        grid_images[0].append(grid_cell_image)

        return grid_images

class ClassifyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

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

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training phase
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (len(train_loader) * train_loader.batch_size))
                pbar.update()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100.0 * correct / total
            print(f"Epoch {epoch + 1}/{num_epochs} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

        # Update learning rate scheduler
        scheduler.step()

    print("Training completed!")

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and grid size
image_folder_path = "./store/images"
labels_folder_path = "./store/labels/txt"
grid_size = (7, 7)

# Create an instance of the dataset
grid_dataset = GridDataset(image_folder_path, labels_folder_path, grid_size)

# Prepare the dataset for classification
img2grid = [[], []]
for i in range(len(grid_dataset)):
    for j in range(len(grid_dataset[i])):
        image, label = grid_dataset[i][j], j
        img2grid[0].extend(image)
        img2grid[1].extend([j] * len(grid_dataset[i][j]))

labels = [0, 1]

# Image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
dataset = ClassifyDataset(img2grid[0], img2grid[1], transform=transform)

# Split the dataset into train, validation, and test sets
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders for training, validation, and test sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of classes and model setup
num_classes = len(set(labels))
model = SmallSeparableCNN(num_classes)
model.to(device)

# Training parameters
num_epochs = 10
learning_rate = 0.001

# Initialize the optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# Save the trained model
save_model(model, "trained_model.pth")
