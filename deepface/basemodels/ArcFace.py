from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
import tensorflow
from tensorflow import keras

import os
from pathlib import Path
import gdown

from deepface.commons import functions



from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#url = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"





# def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'):
# 	base_model = ResNet34()
# 	inputs = base_model.inputs[0]
# 	arcface_model = base_model.outputs[0]
# 	arcface_model = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
# 	arcface_model = keras.layers.Dropout(0.4)(arcface_model)
# 	arcface_model = keras.layers.Flatten()(arcface_model)
# 	arcface_model = keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
# 	embedding = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
# 	model = keras.models.Model(inputs, embedding, name=base_model.name)
# 	new_model= Sequential()
# 	new_model.add(model)
# 	new_model.add(Flatten())
# 	new_model.add(Dropout(0.2))
# 	new_model.add(Dense(256,activation='relu'))
# 	new_model.add(Dropout(0.2))
# 	new_model.add(Dense(151,activation='softmax'))
# 	#---------------------------------------
# 	#check the availability of pre-trained weights

# 	home = functions.get_deepface_home()

# 	file_name = "arcface_weights.h5"
# 	output = home+'/.deepface/weights/'+file_name
# 	# output = 'deepface/Saved_model/my_model_weights.h5' 
# 	# if os.path.isfile(output) != True:

# 	# 	print(file_name," will be downloaded to ",output)
# 	# 	gdown.download(url, output, quiet=False)

# 	#---------------------------------------
	
# 	new_model.load_weights('D:/FYP_code/FYP_MODEL_CODE_API/saved_model8/my_model_weights.h5')
# 	# print(model.summary())
# 	model = Model(new_model.input, new_model.layers[1].output)

# 	return model

def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'):
	base_model = ResNet34()
	inputs = base_model.inputs[0]
	arcface_model = base_model.outputs[0]
	arcface_model = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
	arcface_model = keras.layers.Dropout(0.4)(arcface_model)
	arcface_model = keras.layers.Flatten()(arcface_model)
	arcface_model = keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
	embedding = keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
	model = keras.models.Model(inputs, embedding, name=base_model.name)

	#---------------------------------------
	#check the availability of pre-trained weights


	home = functions.get_deepface_home()

	file_name = "arcface_weights.h5"
	output = home+'/.deepface/weights/'+file_name

	if os.path.isfile(output) != True:

		print(file_name," will be downloaded to ",output)
		gdown.download(url, output, quiet=False)

	#---------------------------------------
	
	model.load_weights(output)

	return model

def ResNet34():

	img_input = tensorflow.keras.layers.Input(shape=(112, 112, 3))

	x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name='conv1_pad')(img_input)
	x = tensorflow.keras.layers.Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer='glorot_normal', name='conv1_conv')(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name='conv1_bn')(x)
	x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name='conv1_prelu')(x)
	x = stack_fn(x)

	model = training.Model(img_input, x, name='ResNet34')

	return model

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
	bn_axis = 3

	if conv_shortcut:
		shortcut = tensorflow.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer='glorot_normal', name=name + '_0_conv')(x)
		shortcut = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_0_bn')(shortcut)
	else:
		shortcut = x

	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn')(x)
	x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
	x = tensorflow.keras.layers.Conv2D(filters, 3, strides=1, kernel_initializer='glorot_normal', use_bias=False, name=name + '_1_conv')(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn')(x)
	x = tensorflow.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

	x = tensorflow.keras.layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
	x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=stride, kernel_initializer='glorot_normal', use_bias=False, name=name + '_2_conv')(x)
	x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn')(x)

	x = tensorflow.keras.layers.Add(name=name + '_add')([shortcut, x])
	return x

def stack1(x, filters, blocks, stride1=2, name=None):
	x = block1(x, filters, stride=stride1, name=name + '_block1')
	for i in range(2, blocks + 1):
		x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
	return x

def stack_fn(x):
	x = stack1(x, 64, 3, name='conv2')
	x = stack1(x, 128, 4, name='conv3')
	x = stack1(x, 256, 6, name='conv4')
	return stack1(x, 512, 3, name='conv5')
