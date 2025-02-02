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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
