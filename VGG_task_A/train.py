from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from config import TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH, CELEBA_PATH

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load CelebA dataset for pretraining
celeba_dataset = datasets.CelebA(root=CELEBA_PATH, split='train', transform=transform, download=False)

class GenderDataset(torch.utils.data.Dataset):
    def __init__(self, celeb_dataset):
        self.dataset = celeb_dataset

    def __getitem__(self, idx):
        image, attr = self.dataset[idx]
        gender = attr[20]  # Gender attribute (0: Male, 1: Female)
        return image, gender.long()

    def __len__(self):
        return len(self.dataset)

celeba_train_dataset = GenderDataset(celeba_dataset)
celeba_trainloader = DataLoader(celeba_train_dataset, batch_size=32, shuffle=True)

# Load your dataset for fine-tuning
comsys_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
train_size = int(0.8 * len(comsys_dataset))
val_size = len(comsys_dataset) - train_size
train_data, val_data = random_split(comsys_dataset, [train_size, val_size])

comsys_trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
comsys_valloader = DataLoader(val_data, batch_size=16, shuffle=False)

def train_and_validate(model, trainloader, valloader, num_epochs, device, patience=3):
    """Train and validate the model with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}')

        # Validation loop
        if valloader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

            # Early stopping logic
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Best model saved with Validation Accuracy: {best_val_accuracy:.2f}%")
            else:
                patience_counter += 1
                print(f"No improvement in validation accuracy. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    return model

def main():
    # Load the pre-trained VGG model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 2)  # Replace the final layer for binary classification

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Step 1: Pretrain on CelebA dataset
    print("Pretraining on CelebA dataset...")
    model = train_and_validate(model, celeba_trainloader, None, num_epochs=20, device=device)

    # Step 2: Fine-tune on Comsys dataset
    print("Fine-tuning on Comsys dataset...")
    model = train_and_validate(model, comsys_trainloader, comsys_valloader, num_epochs=40, device=device, patience=10)

    # Save the fine-tuned model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Fine-tuned model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()