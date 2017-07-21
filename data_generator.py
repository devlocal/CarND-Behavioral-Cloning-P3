import csv
import glob
import logging
import os
from math import ceil
from random import shuffle, sample

import cv2
import numpy as np
import sklearn


class DataGenerator(object):
    IMG_FOLDER = "IMG"
    LOG_FILE_NAME = "driving_log.csv"
    CSV_FIELDS = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']

    SIDE_CAMERA_STEERING_ADJUSTMENT = 0.3

    CAMERAS = {
        "center": {"adjustment": 0.0},
        "left": {"adjustment": SIDE_CAMERA_STEERING_ADJUSTMENT},
        "right": {"adjustment": -SIDE_CAMERA_STEERING_ADJUSTMENT},
    }

    def __init__(self, data_path, batch_size=32, cameras=None, flip=False, sample_frac=None, validation_split=None):
        self._data_path = data_path
        self._batch_size = batch_size
        self._cameras = cameras
        self._flip_images = flip
        self._sample_frac = sample_frac
        self._validation_split = validation_split
        self._train_metadata = []
        self._valid_metadata = []

    def _find_log_files(self):
        """

        :return: iterator
        """
        search_path = os.path.join(self._data_path, os.path.join("**", self.LOG_FILE_NAME))
        log_files = glob.iglob(search_path, recursive=True)
        return log_files

    def _load_metadata_from_csv_record(self, log_file_folder, record):
        for camera in self._cameras or ["center"]:
            abs_camera_image_path = os.path.join(log_file_folder, record[camera])
            steering_angle = float(record["steering"]) + self.CAMERAS[camera]["adjustment"]

            self._train_metadata.append({"file": abs_camera_image_path, "steering": steering_angle, "flip": False})

            if self._flip_images:
                self._train_metadata.append({"file": abs_camera_image_path, "steering": -steering_angle, "flip": True})

    def _load_metadata_from_file(self, file_path):
        logging.info("Loading images from %s", file_path)

        abs_log_file_path = os.path.abspath(file_path)
        log_file_folder = os.path.dirname(abs_log_file_path)

        with open(file_path) as log_file:
            reader = csv.DictReader(log_file, self.CSV_FIELDS)
            for n, row in enumerate(reader):
                self._load_metadata_from_csv_record(log_file_folder, row)

    def _compute_files_size(self):
        """

        :param md: list of tuples (file_path, steering_angle)
        :return:
        """
        size = 0
        for value in self._train_metadata:
            size += os.path.getsize(value["file"])
        return size

    @staticmethod
    def _get_mb_size(size):
        return size / 1024.0 / 1024.0

    def _load_metadata(self):
        log_files = self._find_log_files()
        for log_file_path in log_files:
            self._load_metadata_from_file(log_file_path)

        metadata_size_mb = self._get_mb_size(self._compute_files_size())
        logging.info("%d image(s) found, total size: %.1f MB", len(self._train_metadata), metadata_size_mb)

    def _split_validation(self):
        shuffle(self._train_metadata)
        validation_samples = int(len(self._train_metadata) * self._validation_split)

        self._valid_metadata = self._train_metadata[:validation_samples]
        self._train_metadata = self._train_metadata[:validation_samples]

    def build(self):
        self._load_metadata()

        if self._sample_frac:
            self._train_metadata = sample(self._train_metadata, int(len(self._train_metadata) * self._sample_frac))

        if self._validation_split:
            self._split_validation()

    def _get_steps(self, metadata):
        return int(ceil(len(metadata) / self._batch_size))

    def _build_generator(self, metadata):
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
    def train_steps(self):
        return self._get_steps(self._train_metadata)

    @property
    def valid_steps(self):
        return self._get_steps(self._valid_metadata)

    @property
    def train_generator(self):
        return self._build_generator(self._train_metadata)

    @property
    def valid_generator(self):
        return self._build_generator(self._valid_metadata)
