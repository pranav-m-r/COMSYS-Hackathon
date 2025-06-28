import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from deepface import DeepFace
from dataset import FaceDataset
from config import TRAIN_PATH, BASE_FACE_EMBEDDINGS_PATH, MODEL_SAVE_PATH, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, MODEL_NAME
import numpy as np

# Load base face embeddings
base_face_embeddings = np.load(BASE_FACE_EMBEDDINGS_PATH, allow_pickle=True).item()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # DeepFace expects preprocessed images
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = FaceDataset(TRAIN_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define loss and optimizer
criterion = nn.MSELoss()  # Use MSE loss
optimizer = optim.Adam([], lr=LEARNING_RATE)  # No trainable parameters in DeepFace

# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        # Convert images to numpy arrays for DeepFace
        images_np = [image.numpy().transpose(1, 2, 0) for image in images]

        # Extract embeddings using DeepFace
        embeddings = [DeepFace.represent(img, model_name=MODEL_NAME) for img in images_np]

        # Convert base embeddings to tensors
        base_embeddings = torch.tensor([base_face_embeddings[label.item()] for label in labels])

        # Calculate loss
        loss = criterion(torch.tensor(embeddings), base_embeddings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# Save fine-tuned model (if applicable)
print(f"Training completed.")