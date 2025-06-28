import os

# Model configuration
MODEL_NAME = 'vgg16'
USE_FINE_TUNED = False  # Set to True to use the fine-tuned model, False for the pre-trained model

# Paths
TRAIN_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "train")
VAL_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "val")
CELEBA_PATH = os.path.join("..", "data")
MODEL_SAVE_PATH = f"{MODEL_NAME}_fine_tuned.pth"