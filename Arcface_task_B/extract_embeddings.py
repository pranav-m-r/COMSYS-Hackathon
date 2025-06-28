import os
import numpy as np
from deepface import DeepFace
from config import TRAIN_PATH, BASE_FACE_EMBEDDINGS_PATH, MODEL_NAME

def extract_embedding(image_path):
    """Extract face embedding using DeepFace."""
    embedding = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME)
    return embedding

# Extract embeddings for base faces
base_face_embeddings = {}

for person_id in os.listdir(TRAIN_PATH):
    person_folder = os.path.join(TRAIN_PATH, person_id)
    if os.path.isdir(person_folder):
        base_face_path = os.path.join(person_folder, f"{person_id}.jpg")
        embedding = extract_embedding(base_face_path)
        if embedding is not None:
            base_face_embeddings[person_id] = embedding
    print(f"Processed base face for {person_id}")

# Save embeddings for future use
np.save(BASE_FACE_EMBEDDINGS_PATH, base_face_embeddings)
print(f"Base face embeddings saved to {BASE_FACE_EMBEDDINGS_PATH}")