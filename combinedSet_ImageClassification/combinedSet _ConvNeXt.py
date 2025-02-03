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

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download ImageWoof dataset
def download_imagewoof(url, download_path, extract_path):
    if not os.path.exists(os.path.dirname(download_path)):
        os.makedirs(os.path.dirname(download_path))
    if not os.path.exists(extract_path):
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, download_path)
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")
    else:
        print("Dataset already exists.")

# Paths and dataset URLs
imagewoof_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
imagewoof_root = './data/imagewoof2-160'
fgvc_aircraft_root = './data'
flowers102_root = './data/flowers-102'

# Download ImageWoof dataset
download_imagewoof(imagewoof_url, './data/imagewoof2-160.tgz', imagewoof_root)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load datasets using PyTorch built-in datasets and split options
fgvc_trainval = torchvision.datasets.FGVCAircraft(root=fgvc_aircraft_root, split='trainval', download=True, transform=transform)
fgvc_test = torchvision.datasets.FGVCAircraft(root=fgvc_aircraft_root, split='test', download=True, transform=transform)

flowers_train = torchvision.datasets.Flowers102(root=flowers102_root, split='test', download=True, transform=transform)
flowers_test = torchvision.datasets.Flowers102(root=flowers102_root, split='train', download=True, transform=transform)

# Load ImageWoof dataset using ImageFolder (train and val)
train_dir = os.path.join(imagewoof_root, 'imagewoof2-160/train')
valid_dir = os.path.join(imagewoof_root, 'imagewoof2-160/val')

imagewoof_train = ImageFolder(root=train_dir, transform=transform)
imagewoof_val = ImageFolder(root=valid_dir, transform=transform)

# Function to update targets
def update_targets(dataset, start_label):
    dataset.targets = [label for _, label in dataset]
    unique_labels = set(dataset.targets)

    return dataset, start_label + len(unique_labels)

# Remap labels for each dataset to avoid conflicts
start_label = 0

imagewoof_train, start_label_fgcv = update_targets(imagewoof_train, start_label)
imagewoof_val, _ = update_targets(imagewoof_val, start_label)


class ModifiedFGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self, root, split='trainval', download=False, transform=None, startlabel=0):
        super(ModifiedFGVCAircraft, self).__init__(root=root, split=split, download=download, transform=transform)
        self.startlabel = startlabel
    def __getitem__(self, index):
        image, label = super(ModifiedFGVCAircraft, self).__getitem__(index)
        # Add the scalar to the label when returning
        label += self.startlabel
        return image, label
fgvc_trainval = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='trainval', download=True, transform=transform, startlabel=start_label_fgcv)
fgvc_test = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='test', download=True, transform=transform, startlabel=start_label_fgcv)

fgvc_trainval, start_label_flowers = update_targets(fgvc_trainval, start_label_fgcv)
fgvc_test, _ = update_targets(fgvc_test, start_label_fgcv)


class ModifiedFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, root, split='train', download=False, transform=None, startlabel=0):
        super(ModifiedFlowers102, self).__init__(root=root, split=split, download=download, transform=transform)
        self.startlabel = startlabel
    def __getitem__(self, index):
        image, label = super(ModifiedFlowers102, self).__getitem__(index)
        # Add the scalar to the label when returning
        label += self.startlabel
        return image, label
flowers_train = ModifiedFlowers102(root=flowers102_root, split='test', download=True, transform=transform, startlabel=start_label_flowers)
flowers_test = ModifiedFlowers102(root=flowers102_root, split='train', download=True, transform=transform, startlabel=start_label_flowers)

flowers_train, _ = update_targets(flowers_train, start_label_flowers)
flowers_test, _ = update_targets(flowers_test, start_label_flowers)


# Create train and test datasets
train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=True)


# Create train and test datasets
train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load ConvNeXt model
model = models.convnext_tiny(pretrained=True)

# Modify the classifier head for the combined dataset (212 classes)
num_classes = 212
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Measure FLOPs and parameter count
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=False
    )
print(f"FLOPs: {macs}, Parameters: {params}")


# Train the model
print("\nStarting training...")
train_losses = train_model(model, train_loader, num_epochs=15)

# Evaluate the model
print("\nEvaluating on test set...")
test_accuracy = evaluate_model(model, test_loader)

# Plot training loss
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
