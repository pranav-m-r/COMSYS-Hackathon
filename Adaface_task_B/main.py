import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN
from PIL import Image
import os
import numpy as np
# import cv2
import imgaug.augmenters as iaa
import torch.nn.functional as F
from torchvision import transforms


# =====================
# 1. DISTORTION CORRECTION MODULE (U-Net)
# =====================
def center_crop(tensor, target_tensor):
    _, _, h, w = tensor.size()
    _, _, th, tw = target_tensor.size()
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return tensor[:, :, dh:dh+th, dw:dw+tw]

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Decoder path
        x = self.up1(x3)
        x2_cropped = center_crop(x2, x)
        x = torch.cat([x, x2_cropped], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x1_cropped = center_crop(x1, x)
        x = torch.cat([x, x1_cropped], dim=1)
        x = self.conv2(x)
        return torch.sigmoid(self.outc(x))

# =====================
# 2. ADAFACE MODEL
# =====================
class AdaFace(nn.Module):
    def __init__(self, num_classes, embedding_size=512, margin=0.4):
        super(AdaFace, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)  # Add this
        self.margin = margin
        self.num_classes = num_classes
    
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = F.normalize(self.fc(features), p=2, dim=1)
        logits = self.classifier(embeddings)  # logits for cross-entropy
        return logits, embeddings

# =====================
# 3. DATA LOADER
# =====================
class FaceComDataset(Dataset):
    def __init__(self, image_label_list, transform=None):
        self.image_label_list = image_label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        img_path, label = self.image_label_list[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((256, 256), Image.BILINEAR)
        if self.transform:
            image_np = np.array(image)
            # If using imgaug (for train), call with keyword
            if hasattr(self.transform, 'augment'):
                image_np = self.transform(image=image_np)
                if isinstance(image_np, dict):
                    image_np = image_np["image"]
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            else:
                # For torchvision transforms (test), call directly on PIL image
                image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label

# =====================
# 4. MAIN PIPELINE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and augmentation
train_transform = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CloudLayer(
        intensity_mean=0.5,
        intensity_freq_exponent=2.5,
        intensity_coarse_scale=0.5,
        alpha_min=0.0,
        alpha_multiplier=1.0,
        alpha_size_px_max=8,
        alpha_freq_exponent=2.5,
        sparsity=0.8,
        density_multiplier=1.0
    ),
    iaa.Rain(drop_size=(0.01, 0.05)),
    iaa.GammaContrast((0.5, 2.0))
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 1. Build label_dict from all person folders
train_root = './data/Comys_Hackathon5/Task_B/train'
person_folders = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
label_dict = {person: idx for idx, person in enumerate(person_folders)}

# 2. Collect undistorted (train) and distorted (test) image paths and labels
train_image_label_list = []
test_image_label_list = []

for person in person_folders:
    person_dir = os.path.join(train_root, person)
    # Add undistorted image(s): directly inside person_dir, not in 'distortion'
    for fname in os.listdir(person_dir):
        fpath = os.path.join(person_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            train_image_label_list.append((fpath, label_dict[person]))
    # Add distorted images: inside person_dir/distortion/
    distortion_dir = os.path.join(person_dir, 'distortion')
    if os.path.isdir(distortion_dir):
        for fname in os.listdir(distortion_dir):
            fpath = os.path.join(distortion_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image_label_list.append((fpath, label_dict[person]))

# 3. Create datasets and dataloaders
train_dataset = FaceComDataset(train_image_label_list, transform=train_transform)
test_dataset = FaceComDataset(test_image_label_list, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

num_classes = len(label_dict)
adaface = AdaFace(num_classes=num_classes).to(device)

distortion_corrector = UNet().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# Load weights if available
if os.path.exists("distortion_corrector.pth"):
    distortion_corrector.load_state_dict(torch.load("distortion_corrector.pth", map_location=device))
    print("Loaded distortion_corrector weights.")
if os.path.exists("adaface.pth"):
    adaface.load_state_dict(torch.load("adaface.pth", map_location=device))
    print("Loaded adaface weights.")

# Optimizers
optimizer_unet = optim.Adam(distortion_corrector.parameters(), lr=1e-4)
optimizer_ada = optim.Adam(adaface.parameters(), lr=1e-4)

# =====================
# 5. TRAINING LOOP
# =====================
def train(epochs):
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Step 1: Distortion correction
            corrected = distortion_corrector(images)
            
            # Step 2: Face detection with MTCNN
            faces = []
            for img in corrected:
                img_pil = transforms.ToPILImage()(img.cpu())
                face = mtcnn(img_pil)
                if face is None:
                    faces.append(torch.zeros(3, 160, 160))
                elif face.ndim == 4:
                    faces.append(face[0])
                else:
                    faces.append(face) 
            faces = torch.stack(faces).to(device)
            
            # Step 3: AdaFace training
            logits, embeddings = adaface(faces, labels)
            
            # Use logits for cross-entropy
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backpropagation
            optimizer_unet.zero_grad()
            optimizer_ada.zero_grad()
            loss.backward()
            optimizer_unet.step()
            optimizer_ada.step()
        # Save weights after each epoch
        torch.save(distortion_corrector.state_dict(), "distortion_corrector.pth")
        torch.save(adaface.state_dict(), "adaface.pth")
        print(f"Epoch {epoch+1} completed and weights saved.")
# =====================
# 6. TEST FUNCTION
# =====================
def test(model, adaface, dataloader, device):
    model.eval()
    adaface.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            corrected = model(images)
            faces = []
            for img in corrected:
                img_pil = transforms.ToPILImage()(img.cpu())
                face = mtcnn(img_pil)
                if face is None:
                    faces.append(torch.zeros(3, 160, 160))
                elif face.ndim == 4:
                    faces.append(face[0])
                else:
                    faces.append(face) 
            faces = torch.stack(faces).to(device)
            logits, _ = adaface(faces)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print("Predicted class counts:", torch.bincount(preds))
            print("True class counts:", torch.bincount(labels))
    acc = correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc

# Example usage:
# train(20)
test(distortion_corrector, adaface, test_loader, device)

