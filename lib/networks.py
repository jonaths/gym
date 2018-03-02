#!/usr/bin/env python
from __future__ import print_function

from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, \
    Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop

class Networks(object):
    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution

        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=input_shape)
        cnn_feature = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(state_input)
        cnn_feature = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(cnn_feature)
        cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)

        return model

