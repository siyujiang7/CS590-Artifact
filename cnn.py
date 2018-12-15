from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model
import csv
import keras
#from image_generator import train_generator, test_generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import scipy.misc
from PIL import Image as image
import numpy as np
from numpy import genfromtxt
import re
import matplotlib.pyplot as plt
from keras.models import load_model
def extract_number(string):
	string = string.split('_')
	return int(string[0])

labels = genfromtxt('train_labels.csv', delimiter=',')
print(labels.shape)

train_arr = np.zeros((340,300,300,3))
#load training images
print('loading train data')
imgs = os.listdir('preview/')
imgs = sorted(imgs,key=lambda x: extract_number(x))
# print(imgs)
i = 0
for img in imgs:
	x = load_img('preview/'+img)
	x = img_to_array(x)
	train_arr[i] = x
	i+=1

x_train = train_arr[80:]
y_train = labels[80:]

x_test = train_arr[:80]
y_test = labels[:80]
train_labels = labels
print(y_test)

from keras.optimizers import Adam
model1 = Sequential()
model1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(300,300,3), data_format="channels_last"))
model1.add(Convolution2D(32, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(7, activation='sigmoid'))
learning_rate = 0.01
print(learning_rate)
model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
# prediction = model1.predict(x_test[:2])
# print(prediction)
model1.fit(x_train,y_train, batch_size=26, epochs=10, verbose=1, validation_data = (x_test,y_test))
model1.save("weight6.h5")
prediction = model1.predict(x_test[78:])
print(prediction)