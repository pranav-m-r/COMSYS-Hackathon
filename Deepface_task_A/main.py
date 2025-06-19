from deepface import DeepFace
import os
from tqdm import tqdm

TRAIN_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "train")
VAL_PATH = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "val")

def main():
    # Evaluate the model
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0

    for folder in ['female', 'male']:
        correct_label = 'Man' if folder == 'male' else 'Woman'

        # Validate on the train set
        train_folder_path = os.path.join(TRAIN_PATH, folder)
        for filename in tqdm(os.listdir(train_folder_path), desc=f"Validating Train Set ({folder})"):
            filepath = os.path.join(train_folder_path, filename)
            result = DeepFace.analyze(filepath, actions=['gender'], enforce_detection=False)
            if result[0]['dominant_gender'] == correct_label:
                train_correct += 1
            train_total += 1

        # Validate on the validation set
        val_folder_path = os.path.join(VAL_PATH, folder)
        for filename in tqdm(os.listdir(val_folder_path), desc=f"Validating Val Set ({folder})"):
            filepath = os.path.join(val_folder_path, filename)
            result = DeepFace.analyze(filepath, actions=['gender'], enforce_detection=False)
            if result[0]['dominant_gender'] == correct_label:
                val_correct += 1
            val_total += 1

    # Print the accuracy
    train_accuracy = 100 * train_correct / train_total
    val_accuracy = 100 * val_correct / val_total
    print(f"Train Set Accuracy: {train_accuracy:.2f}%")
    print(f"Val Set Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()