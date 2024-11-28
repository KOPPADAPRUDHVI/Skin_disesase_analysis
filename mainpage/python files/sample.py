import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from joblib import dump
from torchvision import datasets, transforms
from torch import optim

# Paths for training and test sets
train_set_path = r"C:\Users\HP\Skin5\skin-disease-datasaet\train_set"
test_set_path = r"C:\Users\HP\Skin5\skin-disease-datasaet\test_set"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load your skin disease dataset
train_dataset = datasets.ImageFolder(root=train_set_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_set_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pretrained model from timm and modify it for your classification task
model = timm.create_model('vit_base_patch16_224', pretrained=True)  # Updated model name
num_classes = len(train_dataset.classes)  # Update based on your dataset
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 5  # Reduced epochs for faster training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

        # Debugging information to confirm shapes
        print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

        optimizer.zero_grad()
        outputs = model(inputs)

        # Debugging output shape
        print(f"Outputs shape: {outputs.shape}")

        # Ensure loss computation has correct dimensions
        try:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        except RuntimeError as e:
            print(f"RuntimeError during loss computation: {e}")
            break  # Exit the loop to debug the error

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# Save the model state
torch.save(model.state_dict(), "dino_skin_disease_model.pth")
print("Model state saved as 'dino_skin_disease_model.pth'")

# Save model parameters for later use with joblib
model_params = {
    'num_classes': num_classes,
    'state_dict': model.state_dict()
}
dump(model_params, 'dino_skin_disease_model.joblib')
print("Model parameters saved as 'dino_skin_disease_model.joblib'")
