# Pattern Recognition, Neural Networks and Deep Learning

## Deep Discriminant Neural Networks - Task Description
This project is a deep neural network implementation for classifying handwritten digits using the MNIST dataset.  

## Prerequisites
To run the code in this project, you need the following dependencies:
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib
  - You can install these dependencies using pip:
    pip install tensorflow keras numpy matplotlib

## Dataset
 - The MNIST dataset is used in this project, which consists of 70,000 28-by-28 pixel images of handwritten digits. 
 - The dataset is split into 60,000 images for training and 10,000 images for testing. 
 - The images are preprocessed by scaling the pixel values between 0 and 1.

## Model Architecture
The deep neural network model has the following architecture:

- Input layer: Accepts 28x28x1 grayscale images.
- Convolutional layers: Convolutional layers with varying filter sizes, activations, and batch normalization.
- Dropout layers: Added after some convolutional layers to prevent overfitting.
- Fully connected layers: Flatten the feature maps and connect them to dense layers.
- Output layer: Outputs probabilities for each digit class.
- Training:
  - The model is trained using the Adam optimizer and categorical cross-entropy loss. 
  
