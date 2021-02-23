import numpy as np
import os
import sys
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow
from DataSet.hurricane_generator import name_visibility_date_dir_seq_generator, name_visibility_date_dir_data_counts


data_root_path = "./DataSet/ScaledData256-RadM/"
save_path = "./Interpolate/liner/"
weights_path = "./GlowKeras/Model/best_encoder.weights"

save_error_path = "./Experiments/liner_result_error/"

typhoon_name = 'IRMA'

gg = name_visibility_date_dir_seq_generator(root_path=data_root_path,
                            length=3,
                            wbl=[ {'white_list':[typhoon_name]}, {'white_list':['Visible']}, {} ])
counts = name_visibility_date_dir_data_counts(root_path=os.path.join(data_root_path, typhoon_name),
                            black_list=['Invisible'])


glow = Glow()
glow.load_weights(weights_path)
glow.create_norm_list(data_root_path, "./DataSet/norm_factor/gaussian.npz", "./DataSet/norm_factor/max_min.npz")

for i in range(counts - 2):
    sds = next(gg)

    # 请使用Interpolate/common.py中的three_frames_interpolate函数
    sd0 = glow.hurricane_normalize(sds[0])
    sd2 = glow.hurricane_normalize(sds[2])
    
    _sd0 = glow.encoder.predict(sd0)
    _sd2 = glow.encoder.predict(sd2)
    
    sd1i = (_sd0 + _sd2) / 2
    sd1i = glow.decoder.predict(sd1i)
    sd1i = glow.hurricane_undo_normalize(sd1i)
    sd1i = np.where(sd1i<0, 0, sd1i)

    sd1 = sds[1]
    sd1 = glow.hurricane_normalize(sd1)
    sd1 = glow.hurricane_undo_normalize(sd1)

    sd0 = glow.hurricane_undo_normalize(sd0)
    sd2 = glow.hurricane_undo_normalize(sd2)

    tmp = [sd0, sd1, sd1i, sd2]

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    save_name = str(i) + ".jpg"
    glow.nparray_to_image(tmp, os.path.join(save_path, save_name))

    save_error_name = str(i) + "_error" + ".jpg"
    error = np.abs(sd0 - sd2)
    glow.nparray_to_image([error], os.path.join(save_error_path, save_error_name))