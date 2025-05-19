import os
import pandas as pd
from model import RetinopathyCNN
import torch
from torchvision import transforms
from PIL import Image
import random

label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Proliferate_DR",
    4: "Severe"
}

def load_model(path="model.pth", device='cpu'):
    model = RetinopathyCNN(num_classes=5)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict(image_path, model, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    pred_class_id = predicted.item()
    pred_class_name = label_map[pred_class_id]
    return pred_class_id, pred_class_name

if __name__ == "__main__":
    device = 'cpu'  # o 'cuda' si tienes GPU y quieres usarla
    model = load_model(device=device)

    # Leer predict.csv
    df = pd.read_csv('data/predict.csv')

    # Elegir una fila aleatoria
    sample = df.sample(1).iloc[0]

    filename = sample[0]
    true_label = int(sample[1])
    true_label_name = label_map[true_label]

    # Construir ruta a la imagen
    img_base_dir = "colored_images"
    folder = label_map[true_label]
    image_path = os.path.join(img_base_dir, folder, filename)

    # Obtener predicción
    pred_class_id, pred_class_name = predict(image_path, model, device=device)

    print(f"Image: {image_path}")
    print(f"True class: {true_label} ({true_label_name})")
    print(f"Predicted class: {pred_class_id} ({pred_class_name})")
