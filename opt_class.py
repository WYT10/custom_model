import os
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
                
                grid_cell_image = image[cell_top:cell_bottom, cell_left:cell_right, :]
                grid_cell_image = torch.from_numpy(grid_cell_image)
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
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    # def __init__(self, data, labels, transform=None):
    #     self.data = data
    #     self.labels = labels
    #     self.transform = transform

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, index):
    #     grids = []
    #     labels = []

    #     image_dataset = self.data[index]
        
    #     for label_index in range(len(image_dataset)):
    #         grids.extend([image_dataset[label_index]])
    #         labels.extend([label_index] * len(image_dataset[label_index]))
                
    #     transformed_images = [self.transform(grid_image) for grid_image in grids]

    #     return transformed_images, labels

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
        batch_size, num_grids, num_channels, height, width = x.size()

        # Reshape the input to combine the batch and grid dimensions
        x = x.view(batch_size * num_grids, num_channels, height, width)

        # Separate RGB channels and normalize the map
        x = self.rgb_norm(x)

        # Feature extraction
        features = self.features(x)

        # Object detection
        output = self.object_detection(features)

        # Reshape the output to separate the batch and grid dimensions again
        output = output.view(batch_size, num_grids, -1)

        return output

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model in training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            image_list, label_list = batch  # Unpack the image and label lists from the batch
            for images, labels in zip(image_list, label_list):
                for image, label in zip(images, labels):
                    image = image.unsqueeze(0)  # Reshape the image to (batch_size, num_channels, height, width)
                    label = torch.tensor(label).unsqueeze(0)  # Convert the label to a tensor and reshape

                    outputs = model(image)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")


image_folder_path = "./store/images"
labels_folder_path = "./store/labels/txt"
grid_size = (7, 7)

# Create an instance of the dataset
grid_dataset = GridDataset(image_folder_path, labels_folder_path, grid_size)

img2grid = [[], []]
for i in range(len(grid_dataset)):
    image, label = grid_dataset[i]
    img2grid[0].append(image)
    img2grid[1].append(label)

labels = [0, 1]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ClassifyDataset(img2grid, labels, transform=transform)

# print(len(dataset[0][1]))

num_classes = 2
batch_size = 16
num_epochs = 10
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = CustomModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, train_loader, criterion, optimizer, num_epochs)

torch.save(model.state_dict(), 'custom_model.pth')