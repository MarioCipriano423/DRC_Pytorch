import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import RetinopathyCNN
from dataset import get_dataloaders
from utils import save_model, calculate_metrics, plot_training_curves, plot_confusion_matrix

def train_model(epochs=10, batch_size=32, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    print("Loading model...")
    model = RetinopathyCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("Starting train...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        print(f"\n📚 Epoch [{epoch+1}/{epochs}]")
        for inputs, labels in tqdm(train_loader, desc="Training...", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds, target_names=[str(i) for i in range(5)])
        train_losses.append(epoch_loss)
        train_accuracies.append(metrics['accuracy'])

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        for inputs, labels in tqdm(val_loader, desc="Validating...", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(val_labels, val_preds, target_names=[str(i) for i in range(5)])
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_metrics['accuracy'])

        print(f"Final of Epoch {epoch+1}")
        print(f"   Train Loss: {epoch_loss:.4f}, Train Acc: {metrics['accuracy']:.4f}")
        print(f"   Val   Loss: {val_epoch_loss:.4f}, Val   Acc: {val_metrics['accuracy']:.4f}")

    model_name = "DRC-model"
    model_path = "models/"+model_name+".pth"
    save_model(model, model_path)
    print("Model trained and saved as "+model_name)

    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    plot_confusion_matrix(val_metrics['confusion_matrix'], class_names=[str(i) for i in range(5)])

if __name__ == "__main__":
    train_model()
