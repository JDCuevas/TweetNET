
from keras.models import Model, Sequential
from keras.layers import Conv1D, Input, Dense, Reshape, Dropout, Add, MaxPooling1D
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.losses import mean_squared_error, mean_absolute_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def TWEETNET(max_words, num_labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

    return model

def BESTNET(max_words, num_labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.summary()
    
    model.compile(loss='categorical_hinge',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def TWEETCONV(num_tweets, max_words, num_labels):
    model = Sequential()
    model.add(Reshape((32, num_tweets, max_words))) 
    model.add(Conv1D(filters = 300, kernel_size = 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_hinge',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
