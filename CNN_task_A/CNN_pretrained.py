from torchvision.datasets import ImageFolder
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os

train_path = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "train")
test_path = os.path.join("..", "data", "Comys_Hackathon5", "Task_A", "val")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard size for CNNs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)
# print(train_data.class_to_idx)
# print(test_data.class_to_idx)
# print(data.samples)

comsys_trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
comsys_testloader = DataLoader(test_data, batch_size=16, shuffle=True)

train_dataset = datasets.CelebA(root=os.path.join("..", "data"), split='train', transform=transform, download=False)

class GenderDataset(torch.utils.data.Dataset):
    def __init__(self, celeb_dataset):
        self.dataset = celeb_dataset

    def __getitem__(self, idx):
        image, attr = self.dataset[idx]
        gender = attr[20]
        return image, gender.long()
    
    def __len__(self):
        return len(self.dataset)
    
train_gender = GenderDataset(train_dataset)
celeba_trainloader = DataLoader(train_gender, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self, trainloader, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(), lr = 0.001, momentum=0.9)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader, 0):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
        print('Finished Training')

    def test(self, testloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}')

cnn = CNN().to(device)

if os.path.exists("pretrained_weights_long.pth"):
    cnn.load_state_dict(torch.load("pretrained_weights_long.pth", map_location=device))
    print("Loaded pretrained weights from CelebA")
else:
    cnn.fit(celeba_trainloader, 20)
    torch.save(cnn.state_dict(), "pretrained_weights_long.pth")
    print("Pretraining completed. Storing weights ...")

cnn.fit(comsys_trainloader, 30)
cnn.test(comsys_testloader)
