#!/usr/bin/env python

import logging

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.models import Sequential
from keras.optimizers import adam

from basic_logging import setup_basic_logging
from data_generator import DataGenerator

MODEL_FILE_PATH = "model.h5"    # TODO: why h5?

setup_basic_logging()

model = Sequential()

model.add(Cropping2D(cropping=((57, 13), (0, 0)), input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Lambda(lambda x: x / 127.5 - 1.0))


def build_nvidia(m):
    # Build NVidia network
    m.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    m.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    m.add(Dropout(rate=0.5))
    m.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    m.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
    m.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
    m.add(Dropout(rate=0.5))
    m.add(Flatten())
    m.add(Dense(1164))
    m.add(Dense(100))
    m.add(Dense(50))
    m.add(Dense(10))
    m.add(Dense(1))


def build_dense(m):
    m.add(Flatten())
    m.add(Dense(1))


# build_dense(model)
build_nvidia(model)

optimizer = adam(
    # lr=0.0003
)
model.compile(loss='mse', optimizer=optimizer)

logging.info("Training...")

checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

train_generator = DataGenerator(
    data_path="train",
    cameras=["center", "left", "right"],
    flip=True,
    # sample_frac=0.1,
    validation_split=0.2
)
train_generator.build()

valid_generator = train_generator

# valid_generator = DataGenerator("validation", sample_frac=0.01, validation_split=1)
# valid_generator.build()

history = model.fit_generator(
    generator=train_generator.train_generator,
    steps_per_epoch=train_generator.train_steps,
    validation_data=valid_generator.valid_generator,
    validation_steps=valid_generator.valid_steps,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

logging.info("Done.")
