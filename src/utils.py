import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np

def calculate_metrics(y_true, y_pred, target_names=None):
    """
    Calcula y retorna métricas principales para clasificación multiclase.
    
    Args:
        y_true (list or np.array): etiquetas reales.
        y_pred (list or np.array): etiquetas predichas.
        target_names (list): nombres de clases para el reporte.

    Returns:
        dict: con accuracy, reporte y matriz de confusión.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, figsize=(8,6), fontsize=12):
    """
    Grafica la matriz de confusión con colores y etiquetas legibles.
    
    Args:
        cm (np.array): matriz de confusión.
        class_names (list): nombres de clases.
        figsize (tuple): tamaño de la figura.
        fontsize (int): tamaño de texto.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.title('Confusion Matrix', fontsize=fontsize+2)
    plt.show()

def plot_training_curves(train_losses, val_losses=None, train_accuracies=None, val_accuracies=None):
    """
    Grafica las curvas de loss y accuracy durante entrenamiento.
    
    Args:
        train_losses (list): pérdidas en cada epoch o batch de entrenamiento.
        val_losses (list, opcional): pérdidas en validación.
        train_accuracies (list, opcional): accuracies en entrenamiento.
        val_accuracies (list, opcional): accuracies en validación.
    """
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

    plt.show()
