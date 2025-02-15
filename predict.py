import torch
from torchvision import transforms
from PIL import Image
from model import CatDogClassifier
from config import *

def load_trained_model():
    model = CatDogClassifier()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(image_path, model=None):
    if model is None:
        model = load_trained_model()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image)
        prediction = output.squeeze().item()
        return "Dog" if prediction > 0.5 else "Cat"

if __name__ == "__main__":
    test_image_path = input("Enter the path to the image you want to classify: ")
    result = predict_image(test_image_path)
    print(f"The image is predicted to be a: {result}")