import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random

import tensorflow.python.keras.engine.base_layer_utils
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
_tf_uses_legacy_keras = True
# TODO. BatchNormalization is not available in the version of tensorflow I have
# I don't think it's important for the model, it doesn't seem to be used in the original code
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout #, BatchNormalization
# TODO. not important for running, but graphs would have been useful for paper/present
# from tensorflow.python.keras.utils import plot_model


# Load images
train_wildfire_dir = 'data/train/damage/wildfire'
train_hurricane_dir = 'data/train/damage/hurricane'
train_earthquake_dir = 'data/train/damage/earthquake'
train_no_damage_dir_root = 'data/train/no_damage'
val_wildfire_dir = 'data/validate/damage/wildfire'
val_hurricane_dir = 'data/validate/damage/hurricane'
val_earthquake_dir = 'data/validate/damage/earthquake'
val_no_damage_dir = 'data/validate/no_damage'
test_wildfire_dir = 'data/test/damage/wildfire'
test_hurricane_dir = 'data/test/damage/hurricane'
test_earthquake_dir = 'data/test/damage/earthquake'
test_no_damage_dir = 'data/test/no_damage'
no_damage_subs = ['earthquake', 'hurricane', 'wildfire']

image_height = 100
image_width = 100

categories = ['wildfire', 'hurricane', 'earthquake', 'no_damage']


# I hate python not allowing out of order function calls
def load_images(directory, label):
	import os
	ret_data = []
	
	for img_file in os.listdir(directory):
		img_path = os.path.join(directory, img_file)
		try:
			img = cv2.imread(img_path)
			img = cv2.resize(img, (image_height, image_width))
			ret_data.append([img, label])
		except Exception as e:
			continue
	
	return ret_data


def loadXY_Data(wildfireDir, hurricaneDir, earthquakeDir, noDamageDir):
	img_data = []
	print("Loading wildfire images...")
	img_data.extend(load_images(wildfireDir, 0))
	print("Loading hurricane images...")
	img_data.extend(load_images(hurricaneDir, 1))
	print("Loading earthquake images...")
	img_data.extend(load_images(earthquakeDir, 2))
	print("Loading no_damage images...")
	for i in range(3):
		img_data.extend(load_images(noDamageDir + '/' + no_damage_subs[i], 3))
	# Shuffle data
	print("Shuffling data...")
	random.shuffle(img_data)
	# the data apparently needs to be in two numpy arrays
	print("Converting data to numpy arrays...")
	x = []
	y_t = []
	for features, labels in img_data:
		x.append(features)
		y_t.append(labels)
	# Convert X and Y list into array
	x_ret = np.array(x, dtype=float)
	y_ret = np.array(y_t)
	
	# Normalize the data
	print("Normalizing data...")
	for i in range(len(x_ret)):
		x_ret[i] = x_ret[i] / 255.0
	
	return x_ret, y_ret


# Load training data
print("Loading training data...")
x_train,y_train = loadXY_Data(train_wildfire_dir, train_hurricane_dir, train_earthquake_dir, train_no_damage_dir_root)
# create validation set
print("Creating validation set...")
x_val, y_val = loadXY_Data(val_wildfire_dir, val_hurricane_dir, val_earthquake_dir, val_no_damage_dir)
# create test set
print("Creating test set...")
x_test, y_test = loadXY_Data(test_wildfire_dir, test_hurricane_dir, test_earthquake_dir, test_no_damage_dir)


# Completed loading training and validation data
print("Completed loading training and validation data")

# Begin the actual model
print("Creating model...")
model = Sequential()

# model from hurricane-damage-prediction-using-cnn
model.add(Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = x_train.shape[1:]))
model.add(AveragePooling2D(2,2))
model.add(Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(3500, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2000, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 8)

# Save the model
timestampString = 'model' + str(int(time.time()))
model.save(timestampString+'.h5')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.grid()
plt.legend(loc = 'best')

plt.show()

model.evaluate(x_test, y_test)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label = 'Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.grid()
plt.legend(loc = 'best')

plt.show()

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.grid()
plt.legend(loc = 'best')
