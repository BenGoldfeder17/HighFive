from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# Define your dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Example training loop
def fine_tune_model(model, train_loader, num_epochs=5, learning_rate=1e-4):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Prepare your dataset and train
train_dataset = CustomDataset(images, labels, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
fine_tune_model(model, train_loader)
