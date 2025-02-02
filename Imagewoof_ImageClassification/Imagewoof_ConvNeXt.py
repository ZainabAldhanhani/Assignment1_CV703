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

def train_model(model, train_loader, num_epochs):
    model.train()
    train_losses = []
    total_training_time = 0  # To track total training time
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()  # End timing
        epoch_time = end_time - start_time
        total_training_time += epoch_time
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Time: {epoch_time:.2f}s")
        
    print(f"Total Training Time: {total_training_time:.2f}s")
    return train_losses


# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

# Hyperparameters
batch_size = 32
num_epochs = 15
learning_rate = 1e-4
weight_decay = 0.01

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
train_dataloader = DataLoader(imagewoof_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(imagewoof_val, batch_size=32, shuffle=False)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ConvNeXt model (pretrained on ImageNet)
model = models.convnext_base(pretrained=True)

# Modify the classifier head for 10 classes (ImageWoof has 10 classes)
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, 10)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Measure FLOPs and parameter count
with torch.cuda.device(0):  # Ensure the device matches your setup
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False
    )
print(f"FLOPs: {macs}, Parameters: {params}")

# Train the model
print("\nStarting training...")
train_losses = train_model(model, train_dataloader, num_epochs=15)

# Evaluate the model
print("\nEvaluating on test set...")
test_accuracy = evaluate_model(model, test_dataloader)

# Plot training loss
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
