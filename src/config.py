import os

# Dataset path (where the Kaggle dataset should be located)
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Subfolders for train and test (assuming a common structure)
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model output
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'diabetic_model.pth')

# Image size (224x224 as per dataset)
IMAGE_SIZE = (224, 224)

# Classes of diabetic retinopathy
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Random seed
SEED = 42
