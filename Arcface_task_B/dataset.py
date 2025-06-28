import os
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing person folders.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for person_id in os.listdir(root_dir):
            person_folder = os.path.join(root_dir, person_id)
            if os.path.isdir(person_folder):
                base_face_path = os.path.join(person_folder, f"{person_id}.jpg")
                self.data.append((base_face_path, person_id))

                distortion_folder = os.path.join(person_folder, "distortion")
                for distorted_image in os.listdir(distortion_folder):
                    distorted_image_path = os.path.join(distortion_folder, distorted_image)
                    self.data.append((distorted_image_path, person_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label