# AI-Image-Classifier

This repository consists of all the codes and resources which were used to construct a deep learning model capable of classifying images as "Real Image" or "AI Generated". This project was prepared under the Coding Club of IIT Guwahati as one of their 'Even semsester Projects'.

## Project Overview and Dataset 

The model uses Convolutional Neural Networks to learn the features of AI based images and real images, and is capable of distinguishing between the two. 
The dataset utilized for the project was essentially scrapped using Selenium and stored in two different 'train' and 'test' folders, each with 'REAL' and 'FAKE' folders with respective images. 

The dataset consists of 5451 training images (REAL and FAKE combined) and 519 testing images (REAL and FAKE combined). 

## Requirements

The project essentially utilized the following tools: 
1. Python 3.11.0
2. Tensorflow
3. Keras
4. Numpy
5. Matplotlib

## Model Performance

Performance of the model on different metrics:
1. Training accuracy: 0.99193
2. Testing accuracy: 0.77073
3. F1 score: 0.77159
4. ROC AUC score: 0.77141

## Access

You can directly access the model via this link : https://imageclassifier-figwm6jwfksxal84rkb8lf.streamlit.app/
