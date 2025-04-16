import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# === Configuration ===
# Path to your dataset folder (must contain subfolders per class, e.g. "trash/", "recycling/")
IMAGE_FOLDER = 'path_to_image_folder'  # TODO: set this to your data folder
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Preprocessing ===
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Optional: safe loader to skip corrupt images
def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (UnidentifiedImageError, OSError):
        # Return a black image on error
        return Image.new('RGB', (224, 224), 'black')

# === Dataset & DataLoaders ===
# Use ImageFolder to infer class labels from subfolder names
full_dataset = datasets.ImageFolder(
    root=IMAGE_FOLDER,
    transform=PREPROCESS,
    loader=safe_pil_loader  # comment out if you prefer default loader
)
classes     = full_dataset.classes
num_classes = len(classes)
print(f"Detected classes: {classes} (num_classes={num_classes})")

# Split into 80% train, 20% validation
total      = len(full_dataset)
train_size = int(0.8 * total)
val_size   = total - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

# === Model Definition ===
# Simple Flatten → Linear(num_classes) model
data_dim = 3 * 224 * 224
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(data_dim, num_classes)
).to(DEVICE)

# === Training Function ===
def fine_tune(model, train_loader, val_loader, epochs=5, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} Training loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        print(f"Epoch {epoch} Validation loss: {val_loss/len(val_loader):.4f}, "
              f"Accuracy: {correct/total:.4f}")

# === Script Entry Point ===
if __name__ == '__main__':
    fine_tune(model, train_loader, val_loader)
    save_path = 'garbage_classifier.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")
