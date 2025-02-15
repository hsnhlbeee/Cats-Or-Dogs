import torch

# Configuration parameters
TRAINING_DIR = r"C:\Users\Hasan\OneDrive\Desktop\projects\data\training_set\training_set"
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = 'cat_dog_classifier.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'