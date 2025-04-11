import os
import csv
import torch
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

# Folder containing images to classify
image_folder = 'path_to_image_folder'  # Replace with the folder path containing your images
output_csv = 'classified_data.csv'

# Function to classify images based on folder structure
def classify_images_by_folder(image_folder, output_csv):
    # Ensure the output CSV file is created
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])  # Header row

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
                preprocess(image)  # Ensure image is valid for preprocessing

                # Save the classification to the CSV file
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([os.path.join(subfolder, image_name), label])

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

# Function to load dataset from CSV
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

# Run classification based on folder structure if CSV does not exist
if not os.path.exists(output_csv):
    classify_images_by_folder(image_folder, output_csv)

# Load dataset from CSV
dataset = load_dataset_from_csv(image_folder, output_csv, preprocess)

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
        # Add progress bar for validation
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

# Example usage
loaded_model = load_model('garbage_classifier.pth')
test_image_path = 'path_to_test_image.jpg'  # Replace with your test image path
predicted_class = predict(test_image_path, loaded_model, preprocess)
print(f"Predicted class: {predicted_class}")
