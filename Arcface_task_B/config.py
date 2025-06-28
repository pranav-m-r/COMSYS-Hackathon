import os

# Paths
TRAIN_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_B", "train")
TEST_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_B", "test")
MODEL_SAVE_PATH = "fine_tuned_arcface.pth"
BASE_FACE_EMBEDDINGS_PATH = "base_face_embeddings.npy"

# Model configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
THRESHOLD = 0.15  # Cosine similarity threshold for matching
MODEL_NAME = "Facenet"  # DeepFace model name (e.g., "Facenet", "VGG-Face", "OpenFace", "DeepFace")