#!/usr/bin/env python

import logging

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.models import Sequential

from basic_logging import setup_basic_logging
from data_generator import DataGeneratorFactory

MODEL_FILE_PATH = "model.h5"

# Set up logging to stdout
setup_basic_logging()

model = Sequential()
# Crop image
model.add(Cropping2D(cropping=((57, 13), (0, 0)), input_shape=(160, 320, 3)))
# Normalize data
model.add(Lambda(lambda x: x / 127.5 - 1.0))


def build_network(m):
    """
    Builds a modified NVidia network.

    Modifications to the original network:
      1. One more convolutional layer is added;
      2. Two dropout layers are added.

    See also: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

    :param m: Model to add network to
    """

    m.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    m.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    m.add(Dropout(p=0.25))
    m.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    m.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu"))
    m.add(Convolution2D(128, 3, 3, subsample=(1, 1), activation="relu"))
    m.add(Dropout(p=0.25))
    m.add(Flatten())
    m.add(Dense(1164))
    m.add(Dense(100))
    m.add(Dense(50))
    m.add(Dense(10))
    m.add(Dense(1))


# Build the neural network
build_network(model)

# Use MSE loss metric and adam optimizer
model.compile(loss="mse", optimizer="adam")

logging.info("Training...")

# Use checkpointing to save the best performing model
checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH, save_best_only=True)

# Stop when validation loss starts increasing
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

# Build data generator and
#  - use images from all three cameras
#  - augment data by flipping images
#  - separate 30% of data for validation
train_factory = DataGeneratorFactory(
    data_path="train",
    cameras={"center", "left", "right"},
    flip=True,
    validation_split=0.3
)
train_factory.initialize()

# Use part of training data for validation
valid_factory = train_factory

# Train the model at most 10 epochs
history = model.fit_generator(
    generator=train_factory.train_generator,
    samples_per_epoch=train_factory.train_samples,
    validation_data=valid_factory.valid_generator,
    nb_val_samples=valid_factory.valid_samples,
    nb_epoch=10,
    callbacks=[checkpoint, early_stopping]
)

logging.info("Done.")
