import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Set device for training and inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing (resizing, normalization) for images
target_size = (224, 224)
preprocess = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Path to your image folder (must contain subfolders like 'trash', 'recycling')
image_folder = 'path_to_image_folder'  # TODO: replace with your actual path
output_csv = 'classified_data.csv'

# 1) Function to classify images by folder and write CSV of (relative_path, label)
def classify_images_by_folder(folder, csv_path):
    # Write header
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Label'])

    # Iterate subfolders and assign numeric labels
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    for label, sub in enumerate(classes):
        sub_path = os.path.join(folder, sub)
        for fname in os.listdir(sub_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            rel_path = os.path.join(sub, fname)
            try:
                img = Image.open(os.path.join(folder, rel_path)).convert('RGB')
                preprocess(img)  # validate preprocessing
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([rel_path, label])
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")

# 2) Function to load dataset from the generated CSV
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label


def load_dataset_from_csv(folder, csv_file, transform):
    samples = []
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(folder, row['Image'])
            label = int(row['Label'])
            samples.append((img_path, label))
    return CustomDataset(samples, transform)

# Create CSV if missing
def prepare_csv():
    if not os.path.exists(output_csv):
        classify_images_by_folder(image_folder, output_csv)

# Main execution
if __name__ == '__main__':
    # Step 1: Ensure CSV exists
    prepare_csv()

    # Step 2: Load dataset and split
    dataset = load_dataset_from_csv(image_folder, output_csv, preprocess)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Step 3: Determine number of classes
    classes = sorted([d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))])
    num_classes = len(classes)
    print(f"Detected classes {classes} → num_classes = {num_classes}")

    # Step 4: Build model (Flatten → Linear)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * target_size[0] * target_size[1], num_classes)
    ).to(device)

    # Training routine
def fine_tune_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch+1} Training Loss: {running/len(train_loader):.4f}")

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                vloss += loss.item()
        print(f"Epoch {epoch+1} Validation Loss: {vloss/len(val_loader):.4f}")
        model.train()

    # Execute training
    fine_tune_model(model, train_loader, val_loader)

    # Save state dict
    save_name = 'garbage_classifier.pth'
    torch.save(model.state_dict(), save_name)
    print(f"Model saved as '{save_name}'")
