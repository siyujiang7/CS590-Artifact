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
# import matplotlib.pyplot as plt

path = '/home/mike/Desktop/experiment/'
def train(learning_rate,folderName):
	from keras.optimizers import Adam
	model1 = Sequential()
	model1.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(150,150,3), data_format="channels_last"))
	model1.add(Convolution2D(32, (3, 3), activation='relu'))
	model1.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
	model1.add(Dropout(0.25))
	model1.add(Flatten())
	model1.add(Dense(128, activation='relu'))
	model1.add(Dropout(0.5))
	model1.add(Dense(128, activation='relu'))
	model1.add(Dropout(0.5))
	model1.add(Dense(1, activation='sigmoid'))
	print(learning_rate)
	model1.compile(loss='mse',optimizer=Adam(lr=learning_rate), metrics=['mse'])
	model1.save(path+'Experiment5/initi_weight_re.h5')
	del model1
	K.clear_session()
	# model2 = Sequential()
	# model2.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(150,150,3), data_format="channels_last"))
	# model2.add(Convolution2D(32, (3, 3), activation='relu'))
	# model2.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
	# model2.add(Dropout(0.25))
	# model2.add(Flatten())
	# model2.add(Dense(128, activation='relu'))
	# model2.add(Dropout(0.5))
	# model2.add(Dense(1, activation='sigmoid'))
	# model2.compile(loss='binary_crossentropy', optimizer=Adam(lr = learning_rate), metrics=['accuracy'])
	# model2.save(path+folderName+'/initi_weight_bin.h5')
	# del model2
	# K.clear_session()

	# bin_acc = np.zeros(3)
	# i = 0
	# #type 1
	# print("Binary type 1:")
	# x_train = np.load("/home/mike/Desktop/experiment/bin_1/train_img.npy")
	# y_train = np.load("/home/mike/Desktop/experiment/bin_1/train_labels.npy")
	# print(x_train.shape)
	# print(y_train.shape)
	# x_test = np.load("/home/mike/Desktop/experiment/bin_1/test_img.npy")
	# y_test = np.load("/home/mike/Desktop/experiment/bin_1/test_labels.npy")
	# print(x_test.shape)
	# print(y_test.shape)
	# model = load_model(path+folderName+'/initi_weight_bin.h5')
	# model.fit(x_train,y_train, batch_size=50, epochs=1,verbose=1, validation_data=(x_test,y_test),shuffle=True)
	# model.save("/home/mike/Desktop/experiment/"+folderName+"/1.h5")
	# matrice = model.evaluate(x_test,y_test)
	# bin_acc[i] = matrice[1]
	# i+=1
	# del model
	# K.clear_session()

	# #type 2
	# print("Binary type 2:")
	# x_train = np.load("/home/mike/Desktop/experiment/bin_2/train_img.npy")
	# y_train = np.load("/home/mike/Desktop/experiment/bin_2/train_labels.npy")
	# print(x_train.shape)
	# print(y_train.shape)
	# x_test = np.load("/home/mike/Desktop/experiment/bin_2/test_img.npy")
	# y_test = np.load("/home/mike/Desktop/experiment/bin_2/test_labels.npy")
	# print(x_test.shape)
	# print(y_test.shape)
	# model = load_model(path+folderName+'/initi_weight_bin.h5')
	# model.fit(x_train,y_train, batch_size=50, epochs=1,verbose=1, validation_data=(x_test,y_test),shuffle=True)
	# model.save("/home/mike/Desktop/experiment/"+folderName+"/2.h5")
	# matrice = model.evaluate(x_test,y_test)
	# bin_acc[i] = matrice[1]
	# i+=1
	# del model
	# K.clear_session()

	# #type 3
	# print("Binary type 3:")
	# x_train = np.load("/home/mike/Desktop/experiment/bin_3/train_img.npy")
	# y_train = np.load("/home/mike/Desktop/experiment/bin_3/train_labels.npy")
	# print(x_train.shape)
	# print(y_train.shape)
	# x_test = np.load("/home/mike/Desktop/experiment/bin_3/test_img.npy")
	# y_test = np.load("/home/mike/Desktop/experiment/bin_3/test_labels.npy")
	# print(x_test.shape)
	# print(y_test.shape)
	# model = load_model(path+folderName+'/initi_weight_bin.h5')
	# model.fit(x_train,y_train, batch_size=50, epochs=1,verbose=1, validation_data=(x_test,y_test),shuffle=True)
	# model.save("/home/mike/Desktop/experiment/"+folderName+"/3.h5")
	# matrice = model.evaluate(x_test,y_test)
	# bin_acc[i] = matrice[1]
	# i+=1
	# del model
	# K.clear_session()


	# print(bin_acc)
	# np.save("/home/mike/Desktop/experiment/"+folderName+"/acc", bin_acc)

	# reg_mse_test = np.zeros(3)
	# reg_mse_train = np.zeros(3)
	# i = 0
	#type 4
	print("\nRegression type 1:")
	x_train = np.load("/home/mike/Desktop/experiment/re_1/train_img.npy")
	y_train = np.load("/home/mike/Desktop/experiment/re_1/train_labels.npy")
	print(y_train)
	x_test = np.load("/home/mike/Desktop/experiment/re_1/test_img.npy")
	y_test = np.load("/home/mike/Desktop/experiment/re_1/test_labels.npy")
	print(y_test)
	# early_stop= keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0)
	model = load_model(path+'Experiment5/initi_weight_re.h5')
	for layer in model.layers:
		print(layer.get_config())
	history=model.fit(x_train,y_train, batch_size=50, epochs=100, verbose=1, validation_data=(x_test,y_test))
	model.save("/home/mike/Desktop/experiment/Experiment4/"+folderName+".h5")
	np.save("/home/mike/Desktop/experiment/Experiment4/"+folderName+".npy", history.history)
	# matrice = model.evaluate(x_test, y_test)
	# reg_mse_test[i] = matrice[1]
	# matrice = model.evaluate(x_train, y_train)
	# reg_mse_train[i] = matrice[1]
	# i+=1
	del model
	K.clear_session()
	
	# #type 5
	# print("\nRegression type 2:")
	# x_train = np.load("/home/mike/Desktop/experiment/re_2/train_img.npy")
	# y_train = np.load("/home/mike/Desktop/experiment/re_2/train_labels.npy")
	# print(x_train.shape)
	# print(y_train.shape)
	# x_test = np.load("/home/mike/Desktop/experiment/re_2/test_img.npy")
	# y_test = np.load("/home/mike/Desktop/experiment/re_2/test_labels.npy")
	# print(x_test.shape)
	# print(y_test.shape)
	# model = load_model(path+folderName+'/initi_weight_re.h5')
	# history=model.fit(x_train,y_train, batch_size=20, epochs=200,verbose=1, validation_data=(x_test,y_test))
	# model.save("/home/mike/Desktop/experiment/"+folderName+"/5.h5")
	# np.save("/home/mike/Desktop/experiment/"+folderName+"/loss5.npy", history.history)
	# matrice = model.evaluate(x_test, y_test)
	# reg_mse_test[i] = matrice[1]
	# matrice = model.evaluate(x_train, y_train)
	# reg_mse_train[i] = matrice[1]
	# i+=1
	# del model
	# K.clear_session()

	# #type 6
	# print("\nRegression type 3:")
	# x_train = np.load("/home/mike/Desktop/experiment/re_3/train_img.npy")
	# y_train = np.load("/home/mike/Desktop/experiment/re_3/train_labels.npy")
	# print(x_train.shape)
	# print(y_train.shape)
	# x_test = np.load("/home/mike/Desktop/experiment/re_3/test_img.npy")
	# y_test = np.load("/home/mike/Desktop/experiment/re_3/test_labels.npy")
	# print(x_test.shape)
	# print(y_test.shape)
	# model = load_model(path+folderName+'/initi_weight_re.h5')
	# history=model.fit(x_train,y_train, batch_size=20, epochs=200,verbose=1, validation_data=(x_test,y_test))
	# model.save("/home/mike/Desktop/experiment/"+folderName+"/6.h5")
	# np.save("/home/mike/Desktop/experiment/"+folderName+"/loss6.npy", history.history)
	# matrice = model.evaluate(x_test, y_test)
	# reg_mse_test[i] = matrice[1]
	# matrice = model.evaluate(x_train, y_train)
	# reg_mse_train[i] = matrice[1]
	# i+=1
	# del model
	# K.clear_session()

	# reg_mse = np.concatenate((reg_mse_train, reg_mse_test), axis=0)
	# print(reg_mse)
	# np.save("/home/mike/Desktop/experiment/"+folderName+"/mse", reg_mse)