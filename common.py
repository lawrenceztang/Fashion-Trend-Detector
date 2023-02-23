def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()
from collections import defaultdict
import random

import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x

class DataGenerator2(Sequence):
    def __init__(self, x_set, y_set, neighbours, batch_size):
        self.x = x_set
        self.y = y_set
        self.neighbours = neighbours
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        with tf.device('/CPU:0'):
            batch_neighbours = tf.gather(self.x, self.neighbours[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = {"anchors": self.x[idx * self.batch_size:(idx + 1) * self.batch_size], "neighbours": batch_neighbours}
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

directory = "images"
target_size = 32  # Resize the input images.
representation_dim = 512  # The dimensions of the features vector.
projection_units = 128  # The projection head of the representation learner.
num_clusters = 5  # Number of clusters.
k_neighbours = 5  # Number of neighbours to consider during cluster learning.
tune_encoder_during_clustering = False  # Freeze the encoder in the cluster learning.
image_dimension = (256, 256)
input_shape = image_dimension + (3,)

x_data = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels=None,
    image_size=image_dimension
)
x_data = x_data.unbatch()
x_data = np.asarray(list(x_data))
x_data = x_data[:10000]
x_gen = DataGenerator(x_data, 32)

data_preprocessing = keras.Sequential(
    [
        layers.Resizing(target_size, target_size),
        layers.Normalization(),
    ]
)

data_augmentation = keras.Sequential(
    [
        layers.RandomTranslation(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
        ),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.15, fill_mode="nearest"),
        layers.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
        ),
    ]
)

data_preprocessing.layers[-1].adapt(x_data)