# Plant Disease Detection

## Project Overview
This project aims to detect diseases in plants using images of leaves. The goal is to classify plant leaves into healthy or diseased categories to help farmers and researchers monitor plant health efficiently.

## 60% Milestone Work Completed
So far, the following work has been completed:

1. Model Development
    Implemented a Convolutional Neural Network (CNN) using TensorFlow/Keras.
    Model architecture:
    2 Convolutional + MaxPooling layers
    Flatten layer
    Dense (Fully Connected) layers with Dropout
    Softmax output layer for multi-class classification

2. Model Training
    Trained the CNN model on the preprocessed dataset (train and val sets).
    Plotted accuracy and loss curves over 10 epochs.
    Saved the training plots as training_plots.png.

3. Model Evaluation
    Evaluated the trained model on the test set.
    Achieved ~88% test accuracy (subject to dataset size & epochs).
    Saved trained model as plant_disease_model.h5.