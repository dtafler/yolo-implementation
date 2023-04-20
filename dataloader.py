import tensorflow as tf
from utils import preprocess_label, get_bb_coords, draw_bb
import pathlib
import os

class DataLoader:
    def __init__(self, data_dir, grid_len, num_classes, num_boxes):
        self.data_dir = data_dir
        self.grid_len = grid_len
        self.num_classes = num_classes
        self.num_boxes = num_boxes

    # convert file path to label encoding
    def get_label(self, file_path):
        file_path = file_path.numpy().decode("utf-8")
        file_path = pathlib.Path(file_path)
        return preprocess_label(file_path, self.grid_len, self.num_classes, self.num_boxes)

    # find and load image
    def get_image(self, file_path):
        file_path = file_path.numpy().decode("utf-8")
        file_name = file_path.split("/")[-1]
        file_name = file_name[:-3] + "jpg"
        file_path = os.path.join(self.data_dir, "images", file_name)
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        return img

    # get image and label from the label path
    def process_path(self, file_path):
        label = tf.py_function(self.get_label, [file_path], tf.float32)
        img = tf.py_function(self.get_image, [file_path], tf.uint8)
        return img, label

    def create_dataset(self):
        # convert label paths to dataset
        list_ds = tf.data.Dataset.list_files(self.data_dir + "/labels/*", shuffle=False)
        ds = list_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

