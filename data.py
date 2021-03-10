import os
import random
import math

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.preprocessing.image as tfpi
import matplotlib.pyplot as plt
from scipy import ndimage

from utils import tf_pyfunction


class DataLoader():
    def __init__(self, path, img_size=256, channel=3,
                 batch=32, deg=5, train=False, flip=False,
                 rotate=False):
        root = Path(path)
        self._list_ds = tf.data.Dataset.list_files(
            str(root / '*/*')
            )
        self._img_size = img_size
        self._channel = channel
        self._batch = batch
        self._flip = flip
        self._rotate = rotate
        self._deg = deg
        self._AUTOTUNE = tf.data.experimental.AUTOTUNE
        self._train = train
        self.label_dict = {d.stem: i for i, d in enumerate(root.glob('*'))}
    
    def __len__(self):
        return len(list(self._list_ds))

    def _parse_image(self, filename):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    def _img_preprocessing(self, image, label):
        image = tf.image.resize(image, (self._img_size, self._img_size))
        label = tf_pyfunction(label, self._labels, tf.int32)
        return image, label

    def _data_augment(self, image):
        crop_size = int(self._img_size * 0.875)
        image = tf.image.random_crop(image, (crop_size, crop_size, self._channel))
        if self._flip:
            image = tf.image.random_filp_left_right(image)
        if self._rotate:
            image = tf_pyfunction(image, self._random_rotate)
        return image

    def _labels(self, label):
        return self.label_dict[label.numpy().decode('utf-8')]

    def _random_rotate(self, image):
        return ndimage.rotate(image, np.random.uniform(-self._deg, self._deg), reshape=False)

    def prepare(self):
        ds = self._list_ds.map(self._parse_image)
        if self._train:
            ds = ds.shuffle(len(self)//4)
            ds = ds.map(lambda x, y: (self._img_preprocessing(x, y)), num_parallel_calls=self._AUTOTUNE)
            ds = ds.map(lambda x, y: (self._data_augment(x), y), num_parallel_calls=self._AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (self._img_preprocessing(x, y)), num_parallel_calls=self._AUTOTUNE)
        ds = ds.batch(self._batch).prefetch(buffer_size=self._AUTOTUNE)
        return ds

    @staticmethod
    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.savefig('sample.png')


if __name__ == '__main__':
    from utils import gpu_select
    gpu_select()
    root = '/raid/rayxie/datasets/Dog_Cat_dataset/train'
    data = DataLoader(root)
    image, label = next(iter(data.prepare()))
    data.show(image, label)
