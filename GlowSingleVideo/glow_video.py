import numpy as np
import os
import math
import time
import sys
import random
import glob

from PIL import Image
from skvideo import io
from skvideo import datasets

from keras import backend as K
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow


class GlowSingleVideo(Glow):
    def __init__(self, batch_size=32, 
                                    lr=1e-3, 
                                    level=3, 
                                    depth=8, 
                                    vd='validate_data', 
                                    data_shape=(64,64,3), 
                                    dam=None
    ):
        super(GlowSingleVideo, self).__init__(
            batch_size,
            lr,
            level,
            depth,
            vd,
            data_shape,
            dam
        )