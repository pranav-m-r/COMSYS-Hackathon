from torchvision.datasets import ImageFolder
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_path = r"data/Comys_Hackathon5/Task_A/train"
test_path = r"data/Comys_Hackathon5/Task_A/val"

transform = transforms.Compose([
    transforms.Resize([244, 244]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)
print(train_data.class_to_idx)
print(test_data.class_to_idx)
# print(data.samples)

trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
testloader = DataLoader(test_data, batch_size=4, shuffle=True)

## ---- VISUALIZING A BATCH ----------
# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = transforms.functional.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# for images, labels in trainloader:
#     grid = make_grid(images)
#     show(grid)
#     plt.show()
#     break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 58 * 58, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 58 * 58)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, trainloader, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(), lr = 0.001, momentum=0.9)

        for epoch in range(25):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = cnn(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    print(f'{epoch+1}, {i+1}, loss : {running_loss/(i+1)}')

        print('Finished Training')

    def test(self, trainloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

cnn = CNN().to(device)
cnn.train(trainloader, 25)
cnn.test(testloader)
