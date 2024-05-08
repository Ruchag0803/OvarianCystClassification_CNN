# PCOS Detection System

## Overview

This project aims to develop a Polycystic Ovary Syndrome (PCOS) detection system using convolutional neural networks (CNNs) for image classification and image processing techniques for highlighting cysts in medical images. The system is integrated into a web application using Flask, allowing users to upload images for PCOS classification and visualization of cysts.

## Project Structure

- **PCOS:** Contains images used for training and validation of the classification model.
- **static:** Stores CSS and JavaScript files for the web interface.
- **templates:** Contains HTML files defining the web interface.
- **test_data_dir:** Stores images used for testing the PCOS detection system.
- **model1.py:** Python script for creating and training the CNN model using Keras. The model architecture includes layers for image preprocessing, convolution, pooling, and fully connected layers with categorical cross-entropy loss and softmax activation.
- **ModelOvarianCystClassification.h5:** Pre-trained model file generated after training the CNN.
- **app.py:** Main interface file for the web application. Flask is used to integrate the front-end with the trained model, allowing users to upload images, classify PCOS, and visualize cysts.
- **cystMarker.py:** Python script containing functions for highlighting cysts in medical images using image processing techniques like thresholding, morphological operations, and watershed algorithm.

## Technologies Used

- **Convolutional Neural Network (CNN):** Utilized for PCOS classification, employing techniques such as categorical cross-entropy loss and softmax activation for multiclass classification.
- **Flask:** Used as the web framework to create the interface between the front-end and the trained model, allowing users to interact with the PCOS detection system.
- **HTML/CSS/JavaScript:** Utilized for creating the user interface and styling the web application.
- **PIL (Python Imaging Library):** Used for image processing tasks such as resizing and converting images to grayscale.
- **OpenCV:** Employed for advanced image processing tasks such as thresholding, morphological operations, and distance transformation for cyst highlighting.
- **Keras:** Used as a high-level neural networks API for building and training the CNN model.

## Video and Output Images

The project includes a video file demonstrating the PCOS detection system in action. Additionally, output images generated during the process are stored in appropriate directories within the project structure.
Working Video is in working folder

<h3>Non-infected Image</h3>
<img src="https://github.com/Ruchag0803/OvarianCystClassification_CNN/assets/112757983/43c3dfea-3a36-4f2a-b0ae-1d1f0ed760c5" alt="Non-infected Image" width="800" height="500">

<h3>Infected Image</h3>
<img src="https://github.com/Ruchag0803/OvarianCystClassification_CNN/assets/112757983/1a4e745e-5c82-4048-ae8f-3ce158c53707" alt="Infected Image" width="800" height="500">
