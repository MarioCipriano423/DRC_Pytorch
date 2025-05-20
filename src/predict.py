import os
import pandas as pd
from model import RetinopathyCNN
import torch
from torchvision import transforms
from PIL import Image
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

def load_model(path="models/DRC-model.pth", device='cpu'):
    print("Loading model...")
    model = RetinopathyCNN(num_classes=5)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print("Model loaded.")
    return model

def predict(image_path, model, device='cpu'):
    print(f"Loading image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    print("Making prediction...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]  

    pred_class_id = torch.argmax(probs).item()
    pred_class_name = label_map[pred_class_id]
    pred_confidence = probs[pred_class_id].item()

    top3_prob, top3_idx = torch.topk(probs, 3)
    top3_classes = [(label_map[i.item()], p.item()) for i, p in zip(top3_idx, top3_prob)]

    return pred_class_id, pred_class_name, pred_confidence, top3_classes, image

def show_image_with_prediction(image, pred_class_name, pred_confidence, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{pred_class_name} ({pred_confidence*100:.2f}%)", fontsize=14, color='green')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Image with prediction saved in: {save_path}")


if __name__ == "__main__":
    device = 'cpu' 

    print("Starting prediction...")
    model = load_model(device=device)

    print("Reading file predict.csv...")
    df = pd.read_csv('data/predict.csv')

    print("Selecting random sample of the CSV...")
    sample = df.sample(1).iloc[0]

    filename = sample['id_code']
    true_label = int(sample['diagnosis'])
    true_label_name = label_map[true_label]
    save_path = "predictions/p_"+str(filename)+"("+str(true_label)+", "+str(true_label_name)+").png"

    if not filename.endswith('.png'):
        filename += '.png'

    print("Building image path...")
    img_base_dir = "data/colored_images"
    folder = label_map[true_label]
    image_path = os.path.join(img_base_dir, folder, filename)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    pred_class_id, pred_class_name, pred_confidence, top3_classes, image = predict(image_path, model, device=device)

    print("\nResult:")
    print(f"Image: {image_path}")
    print(f"Real class: {true_label} ({true_label_name})")
    print(f"Predicted class: {pred_class_id} ({pred_class_name})")
    print(f"Trust: {pred_confidence:.4f} ({pred_confidence*100:.2f}%)")

    print("\n🥇 Top-3 predictions:")
    for i, (class_name, prob) in enumerate(top3_classes):
        print(f" {i+1}. {class_name} ({prob*100:.2f}%)")

    show_image_with_prediction(image, pred_class_name, pred_confidence,save_path)
