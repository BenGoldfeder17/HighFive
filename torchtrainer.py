import os
import csv
import torch
import pandas as pd  # Needed for parquet support
from torchvision import transforms, datasets 
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing (example: resizing and normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Folder containing images to classify, and output file
image_folder = 'path_to_image_folder'  # Replace with your folder containing images
# Change the extension here to either '.csv' or '.parquet' depending on your needs
output_data_file = 'classified_data.parquet'

# Function to classify images based on folder structure
def classify_images_by_folder(image_folder, output_data_file):
    output_ext = os.path.splitext(output_data_file)[1].lower()
    data_entries = []

    # Iterate through subfolders (e.g., "trash" and "recycling")
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
                # Load and preprocess the image to ensure it's valid
                image = Image.open(image_path).convert('RGB')
                preprocess(image)  # Validate image by running through the transforms
                data_entries.append({'Image': os.path.join(subfolder, image_name), 'Label': label})
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    # Write the collected data to the file based on the file extension
    if output_ext == '.csv':
        with open(output_data_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Image', 'Label'])
            writer.writeheader()
            for row in data_entries:
                writer.writerow(row)
    elif output_ext == '.parquet':
        df = pd.DataFrame(data_entries)
        df.to_parquet(output_data_file, index=False)
    else:
        raise ValueError("Unsupported file extension for output file. Use .csv or .parquet")

# Function to load dataset from either CSV or Parquet
def load_dataset_from_file(image_folder, data_file, transform):
    samples = []
    ext = os.path.splitext(data_file)[1].lower()

    if ext == '.csv':
        with open(data_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = os.path.join(image_folder, row['Image'])
                label = int(row['Label'])
                samples.append((image_path, label))
    elif ext == '.parquet':
        df = pd.read_parquet(data_file)
        for _, row in df.iterrows():
            image_path = os.path.join(image_folder, row['Image'])
            label = int(row['Label'])
            samples.append((image_path, label))
    else:
        raise ValueError("Unsupported data file format. Use .csv or .parquet")

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

# Run classification based on folder structure if the data file does not exist yet
if not os.path.exists(output_data_file):
    classify_images_by_folder(image_folder, output_data_file)

# Load dataset from CSV or Parquet file
dataset = load_dataset_from_file(image_folder, output_data_file, preprocess)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define your model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(3 * 224 * 224, 10)  # 10 output classes: battery, biological, cardboard, etc.
).to(device)

# Training loop
def fine_tune_model(model, train_loader, val_loader, num_epochs=5, learning_rate=1e-4):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Add progress bar for training
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Training Loss: {train_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        model.train()

# Train the model
fine_tune_model(model, train_loader, val_loader)

# Save the trained model
torch.save(model.state_dict(), 'garbage_classifier.pth')
print("Model saved as 'garbage_classifier.pth'.")

# Load the model for inference
def load_model(model_path):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 10)  # Ensure architecture matches
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Inference function
def predict(image_path, model, preprocess):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# Example usage for inference
loaded_model = load_model('garbage_classifier.pth')
test_image_path = 'path_to_test_image.jpg'  # Replace with your test image path
predicted_class = predict(test_image_path, loaded_model, preprocess)
print(f"Predicted class: {predicted_class}")
