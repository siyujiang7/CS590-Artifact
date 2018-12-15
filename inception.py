from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model
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
path = '/Users/siyujiang/Desktop/keras'
labels = genfromtxt('/Users/siyujiang/Desktop/keras/norm_labels1.csv', delimiter=',')
print(labels.shape)

def extract_number(string):
	string = string.split('.')
	return int(string[0])

train_arr = np.zeros((2500,299,299,3))
#load training images
print('loading train data')
imgs = os.listdir(path+'/train_img/')
imgs = sorted(imgs ,key=lambda x: extract_number(x) )
i = 0
for img in imgs:
	x = load_img(path+'/train_img/'+img)
	x = img_to_array(x)
	train_arr[i] = x
	i+=1
train_labels = labels[0:2500]
print(train_arr.shape)
print(train_labels.shape)
train_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True).flow(train_arr,train_labels, batch_size = 50)
print('train data loaded')


test_arr = np.zeros((200,299,299,3))
#load test images
print('loading test data')
imgs = os.listdir(path+'/test_img/')
imgs = sorted(imgs ,key=lambda x: extract_number(x) )
i = 0
for img in imgs:
	x = load_img(path+'/test_img/'+img)
	x = img_to_array(x)
	test_arr[i] = x
	i+=1
test_labels = labels[2500:2700]
print(test_arr.shape)
print(test_labels.shape)
test_generator = ImageDataGenerator().flow(test_arr,test_labels, batch_size = 50)
print('test data loaded')

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet',include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024,activation='relu')(x)
x = (Dropout(0.5))(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(7, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='adam', loss='mse')

# # train the model on the new data for a few epochs
# print('start first train')
# model.fit_generator(generator=train_generator, steps_per_epoch=50, epochs=4,verbose=2,validation_data=test_generator,validation_steps=4)
#model.fit(x=train_arr,y=train_labels, batch_size = 400, epochs=4, verbose=2, validation_data=(test_arr,test_labels), shuffle=True)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

for layer in model.layers:
	layer.trainable = True
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mse')
model.compile(optimizer='adam', loss='mse')
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
print('start second train')
model.fit_generator(generator=train_generator, steps_per_epoch=50, epochs=20,verbose=2,validation_data=test_generator,validation_steps=4)
model.save(path+'/trained_weight3.h5')
#model.fit(x=train_arr,y=train_labels, batch_size = 400, epochs=10, verbose=1, validation_data=(test_arr,test_labels), shuffle=True)



