import torch
import torch.nn as nn
import time

# Define the original model architecture
class OriginalModel(nn.Module):
    def __init__(self, num_classes):
        super(OriginalModel, self).__init__()

        # Separate RGB channels and normalize the map
        self.rgb_norm = nn.BatchNorm2d(3)

        # First convolutional layer for each RGB channel
        self.conv1_r = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_g = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_b = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer to bind the RGB maps
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1)

        # Output layer (fully connected with sigmoid activation)
        self.fc = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Separate RGB channels and normalize the map
        x = self.rgb_norm(x)

        # Split the normalized RGB channels
        r, g, b = torch.split(x, 1, dim=1)

        # First convolutional layers for each RGB channel
        r = self.conv1_r(r)
        g = self.conv1_g(g)
        b = self.conv1_b(b)

        # Concatenate the RGB channels along the channel dimension
        x = torch.cat([r, g, b], dim=1)

        # Second convolutional layer to bind the RGB maps
        x = self.conv2(x)

        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the feature map for the fully connected layer
        x = x.view(x.size(0), -1)

        # Output layer with sigmoid activation
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

# Define the optimized model architecture
class OptimizedModel(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedModel, self).__init__()

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


num_classes = 10000
# Create instances of the original and optimized models
original_model = OriginalModel(num_classes)
optimized_model = OptimizedModel(num_classes)

# Generate a random input image with larger dimensions (60x60 pixels)
input_image = torch.randn(1, 3, 60, 60)

# Run inference for the original model and measure the runtime
start_time = time.time()
original_model.eval()
with torch.no_grad():
    _ = original_model(input_image)
end_time = time.time()
original_model_runtime = end_time - start_time

# Run inference for the optimized model and measure the runtime
start_time = time.time()
optimized_model.eval()
with torch.no_grad():
    _ = optimized_model(input_image)
end_time = time.time()
optimized_model_runtime = end_time - start_time

# Print the runtime of each model
print(f"Original Model Runtime: {original_model_runtime} seconds")
print(f"Optimized Model Runtime: {optimized_model_runtime} seconds")
