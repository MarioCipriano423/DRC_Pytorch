import torch

# Verifica instalación
print("Versión de PyTorch:", torch.__version__)
print("¿CUDA disponible?:", torch.cuda.is_available())

# Crea un tensor simple
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)

# Operación básica
y = x ** 2
print("Resultado:", y)
