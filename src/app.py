import os
from .config import config  # Import relativo dentro del paquete

def main():
    print("hello world")
    os.makedirs(config.TEST_DIR, exist_ok=True)
    print(f"Directorio de prueba creado (o ya existía): {config.TEST_DIR}")
