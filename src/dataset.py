import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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

def get_dataloaders(train_csv='data/train.csv', val_csv='data/test.csv', img_dir='colored_images', batch_size=32):
    """
    Retorna los DataLoaders para entrenamiento y validación
    usando los archivos .csv ya generados.
    """
    train_dataset = RetinopathyDataset(csv_file=train_csv, img_dir=img_dir)
    val_dataset = RetinopathyDataset(csv_file=val_csv, img_dir=img_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
