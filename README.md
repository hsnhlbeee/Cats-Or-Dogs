# Cats vs. Dogs Classifier

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images as either cats or dogs. The model is trained on a labeled dataset and can predict the category of new images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to develop a deep learning model capable of distinguishing between images of cats and dogs. The project consists of data preprocessing, model training, evaluation, and deployment for inference.

## Dataset
The dataset used for this project is the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle, containing 25,000 images of cats and dogs.

## Model Architecture
The CNN model consists of multiple convolutional layers followed by activation functions, max-pooling layers, and fully connected layers. Dropout is applied to prevent overfitting.

### Model Layers:
- **Convolutional Layers**: Extract features from images.
- **ReLU Activation**: Introduces non-linearity.
- **MaxPooling**: Reduces spatial dimensions.
- **Fully Connected Layers**: Classifies images as cats or dogs.
- **Sigmoid Activation**: Outputs a probability score.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Cats-Or-Dogs.git
   cd Cats-Or-Dogs
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset:**
   - Download the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
   - Extract it into a directory named `data/` within the project folder.

## Usage

### Train the Model
Run the following command to train the model:
```bash
python train.py
```

This script:
- Loads and preprocesses the dataset.
- Trains the CNN model.
- Saves the trained model to `cat_dog_classifier.pth`.

### Make Predictions
To classify an image, run:
```bash
python predict.py --image_path path/to/your/image.jpg
```
This will output whether the image is a cat or a dog.

## Results
The model achieves an accuracy of approximately 95% on the validation set. Below are sample predictions:

![Sample Prediction 1](results/sample1.jpg)
*Predicted: Cat*

![Sample Prediction 2](results/sample2.jpg)
*Predicted: Dog*

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
