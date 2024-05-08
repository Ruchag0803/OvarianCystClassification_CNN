import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('ModelOvarianCystClassification.h5')

# Read the image using OpenCV
image = cv2.imread('D:\\TY- SEM 6\\Project\\OvarianCystDetectionCNN\\test_data_dir\\ni3.jpg')

# Convert the image to PIL format
img = Image.fromarray(image)

# Resize the image to match the input size expected by the model
img = img.resize((64, 64))

# Convert the image to a numpy array
img = np.array(img)

# Add an extra dimension to the image array to match the input shape expected by the model
inputImg = np.expand_dims(img, axis=0)

# Make predictions on the input image
predictions = model.predict(inputImg)

# Get the index of the class with the highest probability
predicted_class = np.argmax(predictions)

# Print the predicted class
print(predicted_class)
