#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
	load_and_check_model.py

"""

import os

import tensorflow as tf
from tensorflow import keras


if __name__ == '__main__':

	new_model = tf.keras.models.load_model('saved_model/m5_50ep_4r10')
	'saved_model/m5_30ep_22050_10_real'
	# Check its architecture
	new_model.summary()


	# # Evaluate the restored model
	# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
	# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

	# print(new_model.predict(test_images).shape)
