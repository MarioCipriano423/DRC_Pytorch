import torch
from torchvision import transforms
from PIL import Image
from model import RetinopathyCNN

def load_model(path="model.pth", device='cpu'):
    model = RetinopathyCNN(num_classes=5)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict(image_path, model, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats (puedes ajustar si usas otro dataset)
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Añadir batch dim

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

if __name__ == "__main__":
    model = load_model(device='cpu')  # Cambia a 'cuda' si tienes GPU y quieres usarla
    image_path = "data/sample_image.jpg"
    prediction = predict(image_path, model)
    print(f"Predicted class: {prediction}")
