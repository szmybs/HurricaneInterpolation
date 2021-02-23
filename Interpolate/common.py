import numpy as np
import os
import sys
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.getcwd())


# sds-首尾帧序列　glow-模型实例
def three_frames_interpolation(sds, glow, interpolate_method):
    if len(sds) == 2:
        sd0, sd2 = sds
    elif len(sds) == 3:
        sd0, _, sd2 = sds

    if sd0.dim == 3:
        sd0 = np.expand_dims(sd0, 0)
    if sd2.dim == 3:
        sd2 = np.expand_dims(sd2, 0)
    
    _sd0 = glow.encoder.predict( glow.hurricane_normalize(sd0) )
    _sd2 = glow.encoder.predict( glow.hurricane_normalize(sd2) )
    
    sd1i = interpolate_method(_sd0, _sd2)
    sd1i = glow.decoder.predict(sd1i)
    sd1i = glow.hurricane_undo_normalize(sd1i)
    sd1i = np.where(sd1i<0, 0, sd1i)
    return sd1i