#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    raw_audio_music_classifier_training.py

"""

import numpy as np
import pickle
import os
from glob import glob

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from raw_audio_music_config import params
from raw_audio_music_config import music_class_ids


def m5(num_classes):
    """

    """
    m = Sequential()
    m.add(Conv1D(128,
                 input_shape=[params['AUDIO_LENGTH'], 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m


def get_data(file_list):
    """ helper function to get training and test data


    """
    def load_into(_filename, _x, _y):
        with open(_filename, 'rb') as f:
            audio_element = pickle.load(f)
            _x.append(audio_element['audio'])
            _y.append(int(audio_element['class_id']))

    x, y = [], []
    for filename in file_list:
        load_into(filename, x, y)
    return np.array(x), np.array(y)


class RawAudioClassifier(object):
    
    """docstring for RawAudioClassifier"""
    def __init__(self, parameters=params, class_ids=music_class_ids):
        self._num_classes = len(class_ids)
        self._p = parameters
        self._audio_length = self._p['AUDIO_LENGTH']
        self._data_dir = self._p['OUTPUT_DIR']
        self._sample_rate = self._p['TARGET_SR']
        self._data_train = self._p['OUTPUT_DIR_TRAIN']
        self._data_test = self._p['OUTPUT_DIR_TEST']
        self._model = None

    @property
    def audio_length(self):
        return self._audio_length

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def sample_rate(self):
        return self._sample_rate
    
    def setup_model(self):
        self._model = m5(self._num_classes)

        if self._model is None:
            exit('Something went wrong!!')

        self._model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=[
                  tf.keras.losses.CategoricalCrossentropy(
                      from_logits=False, name='categorical_crossentropy'),
                  'accuracy']))




        print(self._model.summary())

    def train_model(self, model_name='anonymous_model'):

        model_filename = f'saved_model/{model_name}'

        train_files = glob(os.path.join(self._data_train, '**.pkl'))
        x_tr, y_tr = get_data(train_files)
        y_tr = to_categorical(y_tr, num_classes=self._num_classes)

        test_files = glob(os.path.join(self._data_test, '**.pkl'))
        x_te, y_te = get_data(test_files)
        y_te = to_categorical(y_te, num_classes=self._num_classes)

        print('x_tr.shape =', x_tr.shape)
        print('y_tr.shape =', y_tr.shape)
        print('x_te.shape =', x_te.shape)
        print('y_te.shape =', y_te.shape)

        # if the accuracy does not increase over 10 epochs, reduce the learning rate by half.
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
        dots = tfdocs.modeling.EpochDots()
        early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=20)
        batch_size = 128
        history = self._model.fit(x=x_tr,
                  y=y_tr,
                  batch_size=batch_size,
                  epochs=20,
                  verbose=1,
                  shuffle=True,
                  validation_data=(x_te, y_te),
                  callbacks=[reduce_lr, dots, early])

        
        print(f"Saving model to {model_filename}")
        self._model.save(model_filename)

        print("DONE")
        return history
    


if __name__ == '__main__':
    

    # raw = RawAudioClassifier()
    # raw.setup_model()
    # raw.train_model(model_name='m5_30ep_22050_10_real2')
    pass