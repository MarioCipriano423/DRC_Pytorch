import os

DATA_DIR = os.path.join(os.getcwd(), 'data')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_PATH = os.path.join(os.getcwd(), 'models', 'diabetic_model.pth')

IMAGE_SIZE = (224, 224)

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

SEED = 42
