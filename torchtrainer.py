import torch
from torchvision import transforms, datasets  # Updated import for datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image  # Import to handle image loading errors
from tqdm import tqdm  # Import tqdm for progress bars

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing (example: resizing and normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom loader to handle invalid files
def safe_loader(path):
    try:
        return Image.open(path).convert('RGB')  # Ensure all images are RGB
    except Exception as e:
        print(f"Error loading image: {path}, skipping. Error: {e}")
        return None

# Update dataset to use the custom loader
dataset = datasets.ImageFolder(
    root='F:\chive\garbage-dataset',
    transform=preprocess,
    loader=safe_loader  # Use the custom loader
)

# Filter out None entries caused by invalid files
valid_indices = [i for i, (path, _) in enumerate(dataset.samples) if dataset.loader(path) is not None]
dataset.samples = [dataset.samples[i] for i in valid_indices]
dataset.targets = [dataset.targets[i] for i in valid_indices]

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
