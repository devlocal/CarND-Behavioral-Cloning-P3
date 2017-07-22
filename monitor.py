import csv
import glob
import logging
import os
import re
import shutil
from tempfile import mkstemp
from time import sleep

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from basic_logging import setup_basic_logging


class MovieFrames(object):
    """Creates movie clips from recorded image files separately for each camera"""

    CAMERAS = ['center']

    def __init__(self, image_folder, file_type="jpg", fps=60):
        self._image_folder = image_folder
        self._file_type = file_type
        self._fps = fps

    def _make_movie_from_camera(self, camera):
        files = glob.glob(os.path.join(self._image_folder, "{}_*.{}".format(camera, self._file_type)))
        movie_file = os.path.abspath(os.path.join(self._image_folder, "..", "{}.mp4".format(camera)))

        clip = ImageSequenceClip(sorted(files), fps=self._fps)
        clip.write_videofile(movie_file)

    def make_movies(self):
        for camera in self.CAMERAS:
            self._make_movie_from_camera(camera)


class CsvLog(object):
    """Localizes path values in all records of a CSV log file"""

    FIELDS = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    PATH_FIELDS = ['center', 'left', 'right']

    def __init__(self, file_path):
        self._file_path = file_path

    @staticmethod
    def _localize_path(path):
        localized_path = path

        # Do not localize absolute path
        if localized_path.startswith("/"):
            parts = localized_path.split("/")
            assert parts[-2] == "IMG"

            # Combine folder ("IMG") and file name
            localized_path = os.path.join(*parts[-2:])

        return localized_path

    def _localize_path_in_row(self, row):
        updated_row = row.copy()
        for field in self.PATH_FIELDS:
            updated_row[field] = self._localize_path(updated_row[field])
        return updated_row

    def localize_path_fields(self):
        """
        Localizes path values in each record of the file

        :return: number of localized records
        """

        # Create a temp file to store localized CSV records
        tmp_f, temp_file_path = mkstemp(text=True)
        os.close(tmp_f)
        num_localized = 0
        try:
            with open(self._file_path) as src_f, open(temp_file_path, "w") as dst_f:
                reader = csv.DictReader(src_f, self.FIELDS)
                writer = csv.DictWriter(dst_f, self.FIELDS)
                for row in reader:
                    updated_row = self._localize_path_in_row(row)
                    writer.writerow(updated_row)
                    if updated_row != row:
                        num_localized += 1

            # Do not overwrite source file if no changes are necessary
            if num_localized:
                shutil.move(temp_file_path, self._file_path)
            else:
                os.remove(temp_file_path)
        except Exception:
            # Clean up on failure
            os.remove(temp_file_path)
            raise
        return num_localized


class Monitor(object):
    """Monitors a folder, moves recorded images and log file to another location"""

    LOG_FILE_NAME = "driving_log.csv"
    IMG_FOLDER_NAME = "IMG"

    # Monitor moves files after no new images appear in IMAGE_FOLDER
    # within (NUM_ITERATIONS - 1) * CHECK_INTEVAL_SEC interval.
    NUM_ITERATIONS = 4
    CHECK_INTERVAL_SEC = 0.25

    def __init__(self, src_folder="data", dest_folder="train"):
        abs_source_folder = os.path.abspath(src_folder)
        self._dest_folder = os.path.abspath(dest_folder)
        self._source_log_file_path = os.path.join(abs_source_folder, self.LOG_FILE_NAME)
        self._source_image_folder = os.path.join(abs_source_folder, self.IMG_FOLDER_NAME)
        self._dest_root = os.path.abspath(dest_folder.split("/")[0])

    def get_files_count(self):
        # Enumerate files only if log file exists
        if not os.path.isfile(self._source_log_file_path):
            return 0

        try:
            return len(os.listdir(self._source_image_folder))
        except:
            return 0

    def get_folder_num(self):
        # Get all folder names
        folders = [os.path.split(w[0])[-1] for w in os.walk(self._dest_root)]
        # Extract numerical parts at the beginning of each folder name
        number_strings = [re.search("^\d*", f).group(0) for f in folders if f[0].isdigit()]
        # Find maximum number
        numbers = [int(n) for n in number_strings]
        try:
            return max(numbers) + 1
        except ValueError:
            return 1

    def move_resources(self):
        """
        Moves image files and csv log to a new folder in DEST_FOLDER.

        :return: path to the new folder
        """
        # Create new folder
        folder_num = self.get_folder_num()
        folder = os.path.join(self._dest_folder, "{:03}".format(folder_num))

        # Move log file and image folder
        os.mkdir(folder)
        shutil.move(self._source_log_file_path, folder)
        shutil.move(self._source_image_folder, folder)

        # Re-create image folder
        os.mkdir(self._source_image_folder)

        return folder

    def run(self):
        fc = [0] * self.NUM_ITERATIONS
        last_num = 0
        while True:
            num = self.get_files_count()
            fc = fc[1:] + [max(0, self.get_files_count() - last_num)]
            last_num = num

            if fc[0] and not sum(fc[1:]):
                # Move resources
                new_folder = self.move_resources()

                # Localize path
                csv_log_path = os.path.join(new_folder, self.LOG_FILE_NAME)
                num_localized = CsvLog(csv_log_path).localize_path_fields()

                # Make movies
                img_folder_path = os.path.join(new_folder, self.IMG_FOLDER_NAME)
                MovieFrames(img_folder_path).make_movies()

                logging.info("Moved resources to %s, localized path fields in %d rows.", new_folder, num_localized)

            sleep(self.CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    setup_basic_logging()
    logging.info("Starting monitor.")
    Monitor().run()
