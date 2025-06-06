import os

class config:
    ## BASE WORKDIR ##
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # data 
    DATA_DIR = os.path.join(BASE_DIR,'..','data')

    # metrics 
    METRICS_DIR = os.path.join(BASE_DIR,'..','metrics')
    CONFUSION_MATRIX_PATH = os.path.join(METRICS_DIR,'confusion_matrix.png')
    TRAINING_CURVES_PATH = os.path.join(METRICS_DIR,'training_curves.png')
    
    # models 
    MODEL_DIR = os.path.join(BASE_DIR,'..','models')

    # predictions 
    PREDICTIONS_DIR = os.path.join(BASE_DIR,'..','predictions')

    # Testing DIR
    TEST_DIR = os.path.join(BASE_DIR,'..','testing')

    # Another stuff
    CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


    # Train parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    SEED = 42

    # dataset parameters
    IMAGE_SIZE = (224, 224)

    