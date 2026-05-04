# Brain Tumor Detection System using Deep Learning
## Overview
This project is a deep learning-based medical image classification system that detects brain tumors from MRI images. It uses a Convolutional Neural Network (CNN) for classification and Grad-CAM for model interpretability.

The system is deployed using Streamlit to provide an interactive web interface.
## Problem Statement
Early detection of brain tumors is critical for effective treatment. Manual diagnosis from MRI scans is time-consuming and prone to human error. This project aims to automate tumor detection using deep learning techniques.

## Classes
The model classifies MRI images into the following categories:

- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

## Features
- CNN-based MRI image classification
- Grad-CAM heatmap for model explainability
- Streamlit web application interface
- Real-time image prediction
- Confidence score display

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Streamlit
- Matplotlib

## Project Workflow
1. Data preprocessing and augmentation
2. CNN model training
3. Model evaluation
4. Grad-CAM implementation
5. Streamlit web app deployment

## Model Architecture
- Convolutional Neural Network (CNN)
- MaxPooling layers
- Dropout for regularization
- Dense layers for classification

## How to Run This Project

### 1. Install dependencies
pip install -r requirements.txt
streamlit run app.py

Input
MRI brain image (JPG / PNG / JPEG)

Output
Predicted class (tumor type)
Confidence score
Grad-CAM heatmap visualization

Author

Muskan Solanki
Data Science / Machine Learning Project
