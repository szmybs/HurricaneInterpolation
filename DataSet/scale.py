import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.getcwd())   

from DataSet.hurricane_generator import HurricaneGenerator
from DataSet.extract import HurricaneExtraction


class FixedScale(object):
    def __init__(self):
        pass
    
    @classmethod
    def goes_conv2d(self, x, kernel_size):
        if (x.shape[0] != x.shape[1]) or x.shape[0] % kernel_size !=0:
            return

        stride = kernel_size
        x = x / (kernel_size * kernel_size)

        y_shape = ( int(x.shape[0] / kernel_size), int(x.shape[0] / kernel_size) )
        y = np.zeros(y_shape, dtype = np.float32)

        for i in range(y_shape[0]):
            for j in range(y_shape[1]):
                x_start = i * stride
                y_start = j * stride
                y[i, j] = np.sum( x[x_start : x_start+stride, y_start : y_start+stride] )
        
        return y

    @classmethod
    def scale_to_fixed_size(self, x, fixed_size):
        nx = []
        for i in x:
            times = int(i.shape[0] / fixed_size)
            if times <= 1:
                nx.append(i)
            else:
                nx.append( self.goes_conv2d(i, times) )
        return nx        


# name - visibility - date
def goes16_5channels_scale_dir(root_path, save_path, read_data_func):
    name_dirs = HurricaneGenerator.directory_downstream(root_path)

    for name_dir in name_dirs:
        visibility_dirs = HurricaneGenerator.directory_downstream(name_dir)

        for visibility_dir in visibility_dirs:
            date_dirs = HurricaneGenerator.directory_downstream(visibility_dir)

            for date_dir in date_dirs:
                new_path = os.path.relpath(path=date_dir, start=root_path)
                new_path = os.path.join(save_path, new_path)
                if os.path.exists(new_path) == False:
                    os.makedirs(new_path)

                data_list = sorted(os.listdir(date_dir))
                for df in data_list:
                    if os.path.isfile == False:
                        continue
                    dp = os.path.join(date_dir, df)
                    d = read_data_func(dp)
                    #d = HurricaneExtraction.convert_unsigned_to_float(d)
                    dfs = FixedScale.scale_to_fixed_size(d, 32)
                    dfs = (np.asarray(dfs)).astype(np.uint16)
                    dfs = np.rollaxis(dfs, 0, 3)                         # C,H,W -> H,W,C

                    file_save_loc = os.path.join(new_path, os.path.splitext(df)[0])
                    np.save(file_save_loc, dfs)
                    print("save to %s" % (file_save_loc))



if __name__ == "__main__":
    # root_path = "./DataSet/Data/"
    # save_path = "./DataSet/ScaledData32/"

    # if os.path.exists(save_path) ==False:
    #     os.mkdir(save_path)
    # goes16_5channels_scale_dir(root_path, save_path, HurricaneExtraction.read_extraction_data)

    root_path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\NpyData\\"
    save_path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ScaledData32\\"

    if os.path.exists(save_path) ==False:
        os.mkdir(save_path)
    goes16_5channels_scale_dir(root_path, save_path, HurricaneExtraction.read_extraction_data)

    