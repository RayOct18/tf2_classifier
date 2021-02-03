import os
import re

import numpy as np
import tensorflow as tf


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
    

if __name__ == '__main__':
    gpu_select()
    print(os.getenv('CUDA_VISIBLE_DEVICES'))
