import cv2
import os
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math


# Function to load and preprocess image data
def load_and_preprocess_data(image_directory, inpSize=64):
    nonInfectedImages = os.listdir(os.path.join(image_directory, 'notinfected'))
    infectedImages = os.listdir(os.path.join(image_directory, 'infected'))
    dataset = []
    label = []

    for imgName in nonInfectedImages:
        if imgName.split('.')[-1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, 'notinfected', imgName))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((inpSize, inpSize))
            dataset.append(np.array(image))
            label.append(0)

    for imgName in infectedImages:
        if imgName.split('.')[-1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, 'infected', imgName))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((inpSize, inpSize))
            dataset.append(np.array(image))
            label.append(1)

    dataset = np.array(dataset)
    label = np.array(label)
    return dataset, label

# Load and preprocess data
image_directory = 'PCOS/'
dataset, label = load_and_preprocess_data(image_directory)

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# Model building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Using 1 neuron and sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

steps_per_epoch = int(math.ceil(len(x_train) / 16))

# Train the model with data augmentation
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=16), 
                    steps_per_epoch=len(x_train) / 16, 
                    epochs=50, 
                    validation_data=(x_test, y_test))




# Save the model
model.save('BinaryClassificationSigmodial.h5')
