import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader,ConcatDataset
import time
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import os
import urllib.request
import tarfile
import torchvision
from torchvision.datasets import ImageFolder
from collections import Counter
from torch.nn import BatchNorm1d

# Hyperparameters
batch_size = 32
num_epochs = 15  # Increased epochs for better convergence
learning_rate = 1e-4
weight_decay = 1e-4  # Adjusted for AdamW optimizer

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for data preprocessing with stronger augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ConvNeXt input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Flowers102 dataset
flowers_train = datasets.Flowers102(root="./data", split="test", download=True, transform=transform)
flowers_test = datasets.Flowers102(root="./data", split="train", download=True, transform=transform)

# Create data loaders
train_dataloader = DataLoader(flowers_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(flowers_test, batch_size=batch_size, shuffle=False, drop_last=True)

# Modify ConvNeXt-Tiny for 102 classes with optimizations
class ModifiedConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=102, freeze_layers=True, activation="GELU"):
        super(ModifiedConvNeXtTiny, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)

        # Freeze early layers to reduce computational cost if needed
        if freeze_layers:
            for param in self.model.features[:1].parameters():
                param.requires_grad = False

        # Modify the classifier head (Optimized for ≤5% extra FLOPs and training time)
        num_features = self.model.classifier[2].in_features

        # Choose activation function dynamically
        activation_layer = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
            "SiLU": nn.SiLU(),
            "GELU": nn.GELU()  # Default (better for ConvNeXt)
        }[activation]

        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(0.2),  # Reduce dropout to minimize compute overhead
            nn.Linear(num_features, 768),  # Intermediate layer (moderate increase)
            activation_layer,
            BatchNorm1d(768),  # BatchNorm AFTER activation for better stability
            nn.Linear(768, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.model(x)



# Initialize the model
model = ModifiedConvNeXtTiny(num_classes=102).to(device)


# Define optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Define MixUp augmentation
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5)

# Define loss function
criterion = SoftTargetCrossEntropy()



# Measure FLOPs and parameter count
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False
    )
print(f"FLOPs: {macs}, Parameters: {params}")


# Define MixUp augmentation
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5, num_classes=102)


# Train the model
print("\nStarting training...")
train_losses = []
total_training_time = 0  # To track total training time
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()  # Start timing
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply MixUp augmentation
        if mixup_fn:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_loss)
    scheduler.step()
    end_time = time.time()  # End timing
    epoch_time = end_time - start_time
    total_training_time += epoch_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")

print(f"Total Training Time: {total_training_time:.2f}s")

# Evaluate the model
print("\nEvaluating on test set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_accuracy = correct / total * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training loss
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
