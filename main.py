from train import train_model
from predict import predict_image

def main():
    # Train the model
    print("Starting training...")
    model = train_model()
    print("Training completed!")

    # Make predictions
    while True:
        test_image_path = input("\nEnter the path to an image to classify (or 'q' to quit): ")
        if test_image_path.lower() == 'q':
            break
        result = predict_image(test_image_path, model)
        print(f"The image is predicted to be a: {result}")

if __name__ == "__main__":
    main() 