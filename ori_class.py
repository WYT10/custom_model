import torch
import torch.nn as nn

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

num_classes = 10
# Create instances of the original and optimized models
model = OriginalModel(num_classes)

print(model)