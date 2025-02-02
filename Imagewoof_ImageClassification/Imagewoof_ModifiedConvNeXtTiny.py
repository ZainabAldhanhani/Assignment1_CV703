import torch
import torch.optim as optim
from torchvision import datasets, models, transforms
import time  
from ptflops import get_model_complexity_info 
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import urllib.request
import tarfile

# Hyperparameters
batch_size = 32
num_epochs = 15  # Increased epochs for better convergence
learning_rate = 1e-4
weight_decay = 1e-4  

def download_imagewoof(url, download_path, extract_path):
    if not os.path.exists(os.path.dirname(download_path)):
        os.makedirs(os.path.dirname(download_path))  # Create directory if it doesn't exist

    if not os.path.exists(extract_path):  # Check if dataset is already extracted
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, download_path)  # Download the dataset
        print("Download complete. Extracting files...")

        # Extract the .tgz file
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)  # Extract files to the folder
        print("Extraction complete.")
    else:
        print("Dataset already exists.")

# URL for ImageWoof dataset
imagewoof_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
download_path = './data/imagewoof2-160.tgz'
extract_path = './data/imagewoof2-160'  # Path to extract the files

# Download and extract the ImageWoof dataset
download_imagewoof(imagewoof_url, download_path, extract_path)

# Set the directories for train and validation sets
train_dir = os.path.join(extract_path, 'imagewoof2-160/train')
valid_dir = os.path.join(extract_path, 'imagewoof2-160/val')

# Define transformations (resize, normalize, and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (standard for pretrained models)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet stats
])

# Load ImageWoof dataset
imagewoof_train = datasets.ImageFolder(root=train_dir, transform=transform)
imagewoof_val = datasets.ImageFolder(root=valid_dir, transform=transform)

# Create data loaders
train_dataloader = DataLoader(imagewoof_train, batch_size=32, shuffle=True,drop_last=True)
test_dataloader = DataLoader(imagewoof_val, batch_size=32, shuffle=False,drop_last=True)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify ConvNeXt-Tiny for 10 classes
class ModifiedConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedConvNeXtTiny, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        
        # Modify the classifier to match the number of classes
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),  # Add Dropout for regularization
            nn.Linear(num_features, num_classes)  # Output 10 classes
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model
model = ModifiedConvNeXtTiny(num_classes=10).to(device)


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
with torch.cuda.device(0):  # Ensure the device matches your setup
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False
    )
print(f"FLOPs: {macs}, Parameters: {params}")

# Continue with the training and evaluation as described in the earlier code

# Define MixUp augmentation
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5, num_classes=10)

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
