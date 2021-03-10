import os
import re

import numpy as np
import tensorflow as tf
import yaml


def gpu_select():
    mem = []
    smi = os.popen('gpustat').readlines()
    for s in smi:
        try:
            mem.append(eval(s.split('|')[2][:-3]))
        except:
            pass
    mini_mem = np.argmin(mem)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mini_mem)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

def tf_pyfunction(image, func, dtype=tf.float32):
    im_shape = image.shape
    [image, ] = tf.py_function(func, [image], [dtype])
    image.set_shape(im_shape)
    return image
    
def load_config(file):
    with open(file, 'r') as stream:
        config = yaml.load(stream)
    return config


if __name__ == '__main__':
    gpu_select()
    print(os.getenv('CUDA_VISIBLE_DEVICES'))
