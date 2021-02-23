import numpy as np
import os
import sys
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow
from DataSet.quantization import Quantization
from DataSet.hurricane_generator import hurricane_load


weights_path = "./GlowKeras/Model/best_encoder.weights"
save_path = "./Interpolate/"
save_error_path = "./Interpolate/"

f0 = np.expand_dims( hurricane_load("./DataSet/ScaledData256-RadM/IRMA/Visible/2017253/20172531608.npy"), axis=0 )
f2 = np.expand_dims( hurricane_load("./DataSet/ScaledData256-RadM/IRMA/Visible/2017253/20172531648.npy"), axis=0 )
f1 = np.expand_dims( hurricane_load("./DataSet/ScaledData256-RadM/IRMA/Visible/2017253/20172531528.npy"), axis=0 )

glow = Glow()
glow.load_weights(weights_path)
glow.create_norm_list("./DataSet/ScaledData256-RadM/", "./DataSet/norm_factor/gaussian.npz", "./DataSet/norm_factor/max_min.npz")


# 请使用Interpolate/common.py中的three_frames_interpolate函数
sd0 = glow.hurricane_normalize(f0)
sd2 = glow.hurricane_normalize(f2)

_sd0 = glow.encoder.predict(sd0)
_sd2 = glow.encoder.predict(sd2)

sd1i = (_sd0 + _sd2) / 2
sd1i = glow.decoder.predict(sd1i)
sd1i = glow.hurricane_undo_normalize(sd1i)
sd1i = np.where(sd1i<0, 0, sd1i)

sd1 = f1
sd1 = glow.hurricane_normalize(sd1)
sd1 = glow.hurricane_undo_normalize(sd1)

sd0 = glow.hurricane_undo_normalize(sd0)
sd2 = glow.hurricane_undo_normalize(sd2)

sd1p = (sd0 + sd2) / 2

error1 = np.abs(sd1 - sd1i)

# tmp = [sd0, sd1, sd1i, sd1p, sd2]
tmp = [sd0, sd1, sd1i, sd2, error1]

save_name = "one_punch_40.png"
glow.nparray_to_image(tmp, os.path.join(save_path, save_name))

# save_error_name = "one_punch_error_5.png"
# error2 = np.abs(sd1 - sd1p)
# glow.nparray_to_image([error1], os.path.join(save_error_path, save_error_name))