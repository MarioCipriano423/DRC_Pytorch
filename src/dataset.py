import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_SIZE

label_to_folder = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Proliferate_DR",
    4: "Severe"
}

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = self.labels.iloc[idx, 0]
        label = int(self.labels.iloc[idx, 1])
        folder = label_to_folder[label]
        img_path = os.path.join(self.img_dir, folder, img_file)
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
