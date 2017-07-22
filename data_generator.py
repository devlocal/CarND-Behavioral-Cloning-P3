import csv
import glob
import logging
import os
from math import ceil
from random import shuffle, sample, random, seed

import cv2
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn


class DataGeneratorFactory(object):
    """
    Data generator factory, builds training and validation data generators.
    """

    IMG_FOLDER = "IMG"
    LOG_FILE_NAME = "driving_log.csv"
    CSV_FIELDS = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']

    SIDE_CAMERA_STEERING_ADJUSTMENT = 0.2

    CAMERAS = {
        "center": {"adjustment": 0.0},
        "left": {"adjustment": SIDE_CAMERA_STEERING_ADJUSTMENT},
        "right": {"adjustment": -SIDE_CAMERA_STEERING_ADJUSTMENT},
    }

    def __init__(self, data_path, batch_size=32, cameras=None, flip=False, sample_frac=None, validation_split=None):
        """
        :param data_path: path to the root of training data
        :param batch_size: training batch size
        :param cameras: (set) cameras to use, any combination of "center", "left", "right"
        :param flip: True to augment data by flipping images horizontally
        :param sample_frac: (float) training data sampling rate, 1 - do not sample
        :param validation_split: (float) fraction of data to use for validation
        """
        self._data_path = data_path
        self._batch_size = batch_size
        self._cameras = cameras or {"center"}
        self._flip_images = flip
        self._sample_frac = sample_frac
        self._validation_split = validation_split
        self._train_metadata = []
        self._valid_metadata = []

    def _find_log_files(self):
        """
        Recursively finds driving_log.log files.

        :return: iterator
        """
        search_path = os.path.join(self._data_path, os.path.join("**", self.LOG_FILE_NAME))
        log_files = glob.iglob(search_path, recursive=True)
        return log_files

    def _load_metadata_from_csv_record(self, log_file_folder, record):
        """
        Loads metadata from a single CSV record and appends to self._train_metadata

        :param log_file_folder: folder containing the log file, used to build path to image files
        :param record: CSV record
        """
        for camera in self._cameras:
            abs_camera_image_path = os.path.join(log_file_folder, record[camera])
            steering_angle = float(record["steering"]) + self.CAMERAS[camera]["adjustment"]

            self._train_metadata.append({"file": abs_camera_image_path, "steering": steering_angle, "flip": False})

            if self._flip_images:
                self._train_metadata.append({"file": abs_camera_image_path, "steering": -steering_angle, "flip": True})

    def _load_metadata_from_file(self, file_path):
        """
        Loads metadata from a single CSV file, appends data to self._train_metadata

        :param file_path: path to CSV file
        """
        logging.info("Loading images from %s", file_path)

        abs_log_file_path = os.path.abspath(file_path)
        log_file_folder = os.path.dirname(abs_log_file_path)

        with open(file_path) as log_file:
            reader = csv.DictReader(log_file, self.CSV_FIELDS)
            for n, row in enumerate(reader):
                self._load_metadata_from_csv_record(log_file_folder, row)

    def _compute_files_size(self):
        """
        Computes size of discovered image files.

        :return: size in bytes
        """
        size = 0
        for value in self._train_metadata:
            size += os.path.getsize(value["file"])
        return size

    @staticmethod
    def _get_mb_size(size):
        """
        Converts size in bytes to size in megabytes.
        :param size: size in bytes
        :return: (float) size in megabytes
        """
        return size / 1024.0 / 1024.0

    def _load_metadata(self):
        """
        Locates log files and loads metadata.
        """
        log_files = self._find_log_files()
        for log_file_path in log_files:
            self._load_metadata_from_file(log_file_path)

    def _split_validation(self):
        """
        Splits loaded metadata into training and validation data sets.
        """
        shuffle(self._train_metadata)
        validation_samples = int(len(self._train_metadata) * self._validation_split)

        self._valid_metadata = self._train_metadata[:validation_samples]
        self._train_metadata = self._train_metadata[:validation_samples]

    def _visualize(self, file_name=None):
        steering = [md["steering"] for md in self._train_metadata]

        plt.figure()
        n, bins, _ = plt.hist(steering, bins=201)

        plt.xlabel('Steering angle')
        plt.ylabel('Number of samples')

        plt.title("Steering angles distribution")

        if file_name:
            plt.savefig(file_name)
        else:
            plt.show()

        return n, bins

    def _balance_buckets(self, n, bis):
        TRIM_THRESHOLD = 4000
        KEEP_RATE = 2000.0 / 14000.0

        selected = []
        for i in range(len(n)):
            if n[i] > TRIM_THRESHOLD:
                selected.append((bis[i], bis[i + 1]))

        for l, h in selected:
            print("l={}, h={}".format(l, h))

        for i in range(len(self._train_metadata))[::-1]:
            for l, h in selected:
                if l <= self._train_metadata[i]["steering"] <= h:
                    if random() > KEEP_RATE:
                        self._train_metadata.pop(i)
                        break

    def initialize(self, balance=False, rand_seed=None):
        """
        Initializes the factory:
          1. loads metadata
          2. subsamples the data
          3. separates data into training and validation sets

        :param balance: True to balance steering angle buckets by trimming peaks
        :param rand_seed: seed for random numbers generator
        :return:
        """

        # Initialize random number generator
        if rand_seed is not None:
            seed(rand_seed)

        self._load_metadata()

        # Balance data buckets
        n, bis = self._visualize("before.png")
        if balance:
            self._balance_buckets(n, bis)
            self._visualize("after.png")

        metadata_size_mb = self._get_mb_size(self._compute_files_size())
        logging.info("Loaded %d image(s), total size: %.1f MB", len(self._train_metadata), metadata_size_mb)

        # Sample data to speed up training
        if self._sample_frac:
            self._train_metadata = sample(self._train_metadata, int(len(self._train_metadata) * self._sample_frac))
            metadata_size_mb = self._get_mb_size(self._compute_files_size())
            logging.info("Sampled data: %d image(s), total size: %.1f MB", len(self._train_metadata), metadata_size_mb)

        # Split into training and validation data
        if self._validation_split:
            self._split_validation()

    def _get_steps(self, metadata):
        """Returns number of steps (for newer versions of Keras)"""
        return int(ceil(len(metadata) / self._batch_size))

    def _build_generator(self, metadata):
        """Builds a new data generator from metadata"""
        steps = self._get_steps(metadata)

        def generator():
            while True:
                shuffle(metadata)

                total_cnt = 0

                for step in range(steps):
                    start_idx = step * self._batch_size
                    mini_batch = metadata[start_idx:start_idx + self._batch_size]
                    total_cnt += len(mini_batch)

                    images = []
                    angles = []
                    for sample_ in mini_batch:
                        image = cv2.imread(sample_["file"])
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        steering_angle = sample_["steering"]

                        image = np.asarray(image).astype(np.float32)

                        if sample_["flip"]:
                            image = np.fliplr(image)

                        images.append(image)
                        angles.append(steering_angle)

                    X = np.asarray(images)
                    y = np.array(angles)

                    yield sklearn.utils.shuffle(X, y)

                assert total_cnt == len(metadata)

        return generator()

    @property
    def train_samples(self):
        """Returns number of samples in training data set"""
        return len(self._train_metadata)

    @property
    def valid_samples(self):
        """Returns number of samples in validation data set"""
        return len(self._valid_metadata)

    @property
    def train_steps(self):
        """Returns number of steps in training data set (for newer versions of Keras)"""
        return self._get_steps(self._train_metadata)

    @property
    def valid_steps(self):
        """Returns number of steps in validation data set (for newer versions of Keras)"""
        return self._get_steps(self._valid_metadata)

    @property
    def train_generator(self):
        """Builds and returns a training data generator"""
        return self._build_generator(self._train_metadata)

    @property
    def valid_generator(self):
        """Builds and returns a validation data generator"""
        return self._build_generator(self._valid_metadata)
