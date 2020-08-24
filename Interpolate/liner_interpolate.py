import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow
from DataSet.hurricane_generator import name_visibility_date_dir_seq_generator


data_root_path = "./DataSet/ScaledData32/"
save_path = "./Interpolate/liner/"
weights_path = "./GlowKeras/Model/best_encoder.weights"

glow = Glow()
glow.load_weights(weights_path)

glow.create_norm_list(data_root_path, "./DataSet/norm_factor/gaussian.npz", "./DataSet/norm_factor/max_min.npz")


gg = name_visibility_date_dir_seq_generator(root_path=data_root_path,
                            length=3,
                            wbl=[ {'white_list':['LIDIA']}, {'white_list':['Visible']}, {} ])
tmp = []
for _ in range(9):
    sds = next(gg)

    sd0 = glow.hurricane_normalize(sds[0])
    sd2 = glow.hurricane_normalize(sds[2])
    
    sd0 = glow.encoder.predict(sd0)
    sd2 = glow.encoder.predict(sd2)
    
    sd1i = (sd0 + sd2) / 2
    sd1i = glow.decoder.predict(sd1i)
    sd1i = glow.hurricane_undo_normalize(sd1i)

    sd1 = sds[1]
    sd1 = glow.hurricane_normalize(sd1)
    sd1 = glow.hurricane_undo_normalize(sd1)

    tmp.append(sd1)
    tmp.append(sd1i)

img = glow.nparray_to_image(tmp)
save_name = "liner_interpolate.tiff"

if os.path.exists(save_path) == False:
    os.path.mkdir(save_path)

img.save(os.path.join(save_path, save_name))