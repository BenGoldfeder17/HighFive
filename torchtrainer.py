import os
import csv
import torch
from torchvision import transforms, datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm  # for progress display

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing (resizing, normalization) for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Folder containing images (should have 'trash' and 'recycling' subfolders)
image_folder = 'path_to_image_folder'  # Replace with the actual folder path
output_csv = 'classified_data.csv'

# Function to classify images by folder and save their labels to CSV
def classify_images_by_folder(image_folder, output_csv):
    # Create CSV file with header if it does not exist
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])
    # Iterate through both subfolders and assign labels
    for label, subfolder in enumerate(['trash', 'recycling']):
        subfolder_path = os.path.join(image_folder, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Subfolder '{subfolder}' not found in {image_folder}, skipping.")
            continue
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {image_name}")
                continue
            try:
                image = Image.open(image_path).convert('RGB')
                # Run the image through preprocessing to validate it
                preprocess(image)
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([os.path.join(subfolder, image_name), label])
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

# Function that loads dataset from a CSV file
def load_dataset_from_csv(image_folder, csv_file, transform):
    samples = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = os.path.join(image_folder, row['Image'])
            label = int(row['Label'])
            samples.append((image_path, label))
    # Custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            image_path, label = self.samples[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, label
    return CustomDataset(samples, transform)

# If CSV does not exist, create it by classifying image folder structure
if not os.path.exists(output_csv):
    classify_images_by_folder(image_folder, output_csv)

# Create dataset and split into training and validation subsets
dataset = load_dataset_from_csv(image_folder, output_csv, preprocess)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the correct model architecture matching the saved state_dict.
# Here, we assume the model is a sequential model with a Flatten layer followed by a Linear layer.
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(3 * 224 * 224, 10)  # 10 output classes (adjust as needed)
).to(device)

# Training loop (runs synchronously)
def fine_tune_model(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-4):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} Training Loss: {avg_train_loss:.4f}")
        
        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss:.4f}")
        model.train()

# Train the model synchronously
fine_tune_model(model, train_loader, val_loader)

# Save the trained model state dict
torch.save(model.state_dict(), 'garbage_classifier.pth')
print("Model saved as 'garbage_classifier.pth'.")
