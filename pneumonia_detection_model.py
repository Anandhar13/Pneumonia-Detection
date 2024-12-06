import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Dropout , BatchNormalization , Flatten , MaxPool2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os
import cv2
from keras.optimizers import Adam #adam optimizers
import scipy

class PneumoniaDetectionModel:

    def __init__(self,input_shape=(256,256,3),learning_rate=0.001):
        """
         INtialize the model with the above input shape and learning rate

        :param input_shape: dimension of ininput shape in tuple data structure
        :param learning_rate: learning rate for optimizer
        """

        self.input_shape=input_shape
        self.learning_rate=learning_rate

        self.model = self._build_model() #build the CNN model

    def _build_model(self):
        model = Sequential()
        """
        Bulid and complie the CNN mode
        :returns the complied sequential model
        """

        # First convolutional layer
        # Adds 32 filters of size 3x3 with ReLU activation
        # Input layer expects images of shape 256x256x3
        model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())  # Normalize activations to speed up training
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))  # Reduce spatial dimensions

        # Second convolutional layer
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))  # Prevent overfitting
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Third convolutional layer
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Fourth convolutional layer
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Increased dropout to reduce overfitting
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Fifth convolutional layer
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # Flatten layer
        # Flattens the output from the convolutional layers into a 1D vector
        model.add(Flatten())

        # Fully connected (dense) layers
        model.add(Dense(units=128, activation='relu'))  # Dense layer with 128 neurons
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model with Adam optimizer and binary cross-entropy loss
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='binary_crossentropy',  # Suitable for binary classification
                      metrics=['accuracy'])  # Track accuracy during training
        return model

    def train(self, train_generator, val_generator, steps_per_epoch, validation_steps, epochs=50):
        # trains history object containing loss and accuracy metrics
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps
        )
        return history

    def evaluate(self, generator, steps): #evaluate the mode on the provided data generators
        return self.model.evaluate(generator, steps=steps)

    def save_model(self, path): #save the trained model
        self.model.save(path)
