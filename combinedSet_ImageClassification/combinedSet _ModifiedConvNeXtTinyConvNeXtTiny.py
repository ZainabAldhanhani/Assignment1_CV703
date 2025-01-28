import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import urllib.request
import tarfile

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

# Load datasets
imagewoof_train = ImageFolder(os.path.join(imagewoof_root, 'imagewoof2-160/train'), transform=transform)
imagewoof_val = ImageFolder(os.path.join(imagewoof_root, 'imagewoof2-160/val'), transform=transform)
fgvc_trainval = torchvision.datasets.FGVCAircraft(root=fgvc_aircraft_root, split='trainval', download=True, transform=transform)
fgvc_test = torchvision.datasets.FGVCAircraft(root=fgvc_aircraft_root, split='test', download=True, transform=transform)
flowers_train = torchvision.datasets.Flowers102(root=flowers102_root, split='test', download=True, transform=transform)
flowers_test = torchvision.datasets.Flowers102(root=flowers102_root, split='train', download=True, transform=transform)

# Custom dataset classes
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, startlabel):
        self.dataset = dataset
        self.startlabel = startlabel
        self.remapped_targets = [label + startlabel for _, label in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, self.remapped_targets[index]

# Remap and combine datasets
start_label = 0
imagewoof_train = RemappedDataset(imagewoof_train, start_label)
start_label += len(set(imagewoof_train.remapped_targets))
fgvc_trainval = RemappedDataset(fgvc_trainval, start_label)
start_label += len(set(fgvc_trainval.remapped_targets))
flowers_train = RemappedDataset(flowers_train, start_label)
start_label += len(set(flowers_train.remapped_targets))
imagewoof_val = RemappedDataset(imagewoof_val, 0)
fgvc_test = RemappedDataset(fgvc_test, len(set(imagewoof_train.remapped_targets)))
flowers_test = RemappedDataset(flowers_test, len(set(imagewoof_train.remapped_targets)) + len(set(fgvc_trainval.remapped_targets)))

# Create train and test datasets
train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=True)


# Modify ConvNeXt-Tiny for 102 classes
class ModifiedConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=212):
        super(ModifiedConvNeXtTiny, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        
        # Modify the classifier to match the number of classes
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),  # Add Dropout for regularization
            nn.Linear(num_features, num_classes)  # Output 102 classes
        )

    def forward(self, x):
        return self.model(x)


# Load ConvNeXt model
model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

# Modify the classifier head for the combined dataset (212 classes)
num_classes = len(set(imagewoof_train.remapped_targets)) + len(set(fgvc_trainval.remapped_targets)) + len(set(flowers_train.remapped_targets))
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Define MixUp augmentation
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5, num_classes=num_classes)


# Define loss function
criterion = SoftTargetCrossEntropy()


# Train the model
print("\nStarting training...")
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
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

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluate the model
print("\nEvaluating on test set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
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