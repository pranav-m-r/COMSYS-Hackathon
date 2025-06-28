import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FaceDataset
from config import TEST_PATH, BASE_FACE_EMBEDDINGS_PATH, THRESHOLD, BATCH_SIZE, MODEL_NAME
import torch

# Load base face embeddings
base_face_embeddings = np.load(BASE_FACE_EMBEDDINGS_PATH, allow_pickle=True).item()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # DeepFace expects preprocessed images
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load test dataset
test_dataset = FaceDataset(TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test matching and calculate accuracy
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        # Convert images to numpy arrays for DeepFace
        images_np = [image.numpy().transpose(1, 2, 0) for image in images]

        # Extract embeddings using DeepFace
        embeddings = [DeepFace.represent(img, model_name=MODEL_NAME) for img in images_np]

        for embedding, label in zip(embeddings, labels):
            best_match = None
            best_score = float("inf")

            # Match with base embeddings
            for person_id, base_embedding in base_face_embeddings.items():
                score = cosine(embedding, base_embedding)
                if score < best_score and score < THRESHOLD:
                    best_score = score
                    best_match = person_id

            if best_match == label.item():  # Correct match
                correct += 1
            elif best_match == "new_face":  # Add new face to the database
                base_face_embeddings[label.item()] = embedding
                print(f"Added new face: {label.item()}")
            total += 1

# Save updated embeddings
np.save(BASE_FACE_EMBEDDINGS_PATH, base_face_embeddings)
print(f"Updated base face embeddings saved to {BASE_FACE_EMBEDDINGS_PATH}")

accuracy = 100 * correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy:.2f}%")