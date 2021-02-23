import numpy as np
import os
import sys
import scipy.stats as stats
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import test_glow_factory
from DataSet.hurricane_generator import name_visibility_date_dir_generator, name_visibility_date_dir_data_counts


def truncated_normal_sample(lower, upper, size=1):
    rv = stats.truncnorm(lower, upper)
    r = rv.rvs(size=size)
    return r


# 输入是一个glow模型的实例
def compute_lower_upper_per_dim(glow):
    cache_path = './Experiments/truncated_normal_range.npz'
    if os.path.exists(cache_path)  == True:
        r = np.load(cache_path)
        return r['max'].tolist(), r['min'].tolist()

    data_root_path = "./DataSet/ScaledData256-RadM/"

    gg = name_visibility_date_dir_generator(root_path=data_root_path, 
                                        batch_size=1,
                                        wbl=[ {'black_list':[glow.validate_seperation]}, {'white_list':['Visible']}, {} ])
    gc = name_visibility_date_dir_data_counts(data_root_path, black_list=[glow.validate_seperation, 'Invisible'])

    high, low = None, None
    for _ in range(gc):
        gdx = next(gg)
        gdx = glow.hurricane_normalize(gdx)

        lv = glow.encoder.predict(gdx)

        if high is None:
            high = lv
        if low is None:
            low = lv
        
        high = np.where(high<lv, lv, high)
        low = np.where(low>lv, lv, low)

    np.savez(file=cache_path, max=np.squeeze(high), min=np.squeeze(low))
    compute_lower_upper_per_dim(glow)


glow = test_glow_factory()
MAX, MIN = compute_lower_upper_per_dim(glow)


# too slowwwwwwwwwwwwwwwwwwwwwww !
def new_glow_sample():
    lv = []
    for i in range(len(MAX)):
        upper = MAX[i]
        lower = MIN[i]
        r = truncated_normal_sample(lower, upper)
        lv.append(r)
    return np.squeeze(np.asarray(lv))

def new_glow_sample2():
    upper = max(MAX)
    lower = min(MIN)
    r = truncated_normal_sample(lower, upper, len(MAX))
    return r


save_path = "./Experiments/truncated_normal_sample/"

for i in range(20):
    group = []
    for j in range(5):
        lv = new_glow_sample()
        lv = np.expand_dims(lv, axis=0)
        s = glow.decoder.predict(lv)
        s = glow.hurricane_undo_normalize(s)
        s = np.where(s<0, 0, s)
        group.append(s)
    
    save_name = str(i) + ".jpg"
    glow.nparray_to_image(group, os.path.join(save_path, save_name))



