import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch
from .config import config

class MetricsEvaluator:
    def __init__(self, y_true, y_pred, target_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = target_names

    def calculate_metrics(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        report = classification_report(self.y_true, self.y_pred, target_names=self.target_names, zero_division=0)
        cm = confusion_matrix(self.y_true, self.y_pred)

        return {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def plot_confusion_matrix(cm, class_names, figsize=(8,6), fontsize=12):

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)
        plt.title('Confusion Matrix', fontsize=fontsize+2)
        plt.tight_layout()
        plt.savefig(config.CONFUSION_MATRIX_PATH)  
        plt.close()

def plot_training_curves(train_losses, val_losses=None, train_accuracies=None, val_accuracies=None):

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    if train_accuracies:
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
    if val_accuracies:
        plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVES_PATH)  
    plt.close()

def save_model(model, path):

    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")
