
from keras.models import Model, Sequential
from keras.layers import Conv2D, Input, Dense, Reshape, Dropout, Add
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.losses import mean_squared_error, mean_absolute_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def TWEETNET():
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

    return model
