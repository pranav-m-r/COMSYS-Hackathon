from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import torch
from config import TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def evaluate_model(model, dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    # Load the fine-tuned VGG model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 2)  # 2 classes: male and female

    print(f"Loading fine-tuned model from {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate on the train set
    train_data = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=False)
    train_accuracy = evaluate_model(model, trainloader, device)
    print(f"Train Set Accuracy: {train_accuracy:.2f}%")

    # Evaluate on the validation set
    val_data = datasets.ImageFolder(VAL_PATH, transform=transform)
    valloader = DataLoader(val_data, batch_size=32, shuffle=False)
    val_accuracy = evaluate_model(model, valloader, device)
    print(f"Validation Set Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()