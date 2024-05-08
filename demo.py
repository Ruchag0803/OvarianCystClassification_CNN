import cv2
import os
from PIL import Image
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
image_directory = 'PCOS/'

nonInfectedImages = os.listdir(image_directory + 'notinfected/')
infectedImages = os.listdir(image_directory + 'infected/')
dataset = []
label = []
inpSize=64

for i, imgName in enumerate(nonInfectedImages):
    if imgName.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'notinfected/' + imgName)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((inpSize, inpSize))
        dataset.append(np.array(image))
        label.append(0)

for i, imgName in enumerate(infectedImages):
    if imgName.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'infected/' + imgName)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((inpSize, inpSize))
        dataset.append(np.array(image))
        label.append(1)


dataset=np.array(dataset)
label=np.array(label)

# print(dataset)
# print(len(label))
# print(label)

# Reshape=(n,image_width,image_height,n_channel)

x_train,x_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)
# print("x_train : ",x_train.shape) # shape of x_train dataset
# print("y_train : ",y_train.shape)
# print("x_test : ",x_test.shape)
# print("y_test : ",y_test.shape)


#Normalizing data: 

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)
#Model building : 

model= Sequential()
model.add(Conv2D(32,(3,3), input_shape=(inpSize,inpSize,3)))  # Conv2D is the class representing a 2D convolutional layer
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Binary CrossEntropy=1, sigmoid
# Categorical Cross Entropy = 2,softmax


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)
model.save('CCEOvarianCystClassification.h5')
