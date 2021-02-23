import numpy as np
import os
import sys
import math

if __name__ == "__main__":
    sys.path.append(os.getcwd())   

from DataSet.hurricane_generator import HurricaneGenerator
from DataSet.extract import HurricaneExtraction
from DataSet.conv2d import conv2d

class FixedScale(object):
    def __init__(self):
        pass
    
    
    @classmethod
    def goes_conv1d(self, x, fixed_size):

        stride = round(x.shape[0] / fixed_size)
        if stride <= 1:
            return x[:fixed_size]

        fs = np.zeros(fixed_size, dtype = np.float32)

        conv = int((x.shape[0] - fixed_size) / (stride - 1))
        conv_size = conv * stride

        for i in range(fs.shape[0]):
            start = i * stride

            if i < conv:
                fs[i] = np.sum( x[start : start+stride] ) / stride
            else:
                fs[i] = x[i * stride  if  start < conv_size  else  conv_size + int((start - conv_size)/stride)]
        
        return fs


    @classmethod
    def goes_conv2d(self, x, fixed_size):
        
        stride_x, stride_y = round(x.shape[0] / fixed_size[0]), round(x.shape[1] / fixed_size[1])
        fs = np.zeros(fixed_size, dtype = np.float32)
        
        conv_x = 1 if stride_x<=1 else int((x.shape[0] - fixed_size[0]) / (stride_x - 1))
        conv_y = 1 if stride_y<=1 else int((x.shape[1] - fixed_size[1]) / (stride_y - 1))

        conv_size_x, conv_size_y = conv_x * stride_x, conv_y * stride_y

        scale = stride_x * stride_y

        for i in range(fs.shape[0]):
            for j in range(fs.shape[1]):
                x_start = i * stride_x
                y_start = j * stride_y

                if i < conv_x and j < conv_y:
                    fs[i, j] = np.sum( x[x_start : x_start+stride_x, y_start : y_start+stride_y] ) / scale
                else:
                    tx = i * stride_x  if  x_start < conv_size_x  else  conv_size_x + int((x_start - conv_size_x)/stride_x)
                    ty = j * stride_y  if  y_start < conv_size_y  else  conv_size_y + int((y_start - conv_size_y)/stride_y)
                    fs[i, j] = x[tx, ty]
        
        return fs


    @classmethod
    def goes_conv1d_2(self, x, fixed_size):
        stride = round(x.shape[0] / fixed_size)
        if stride <= 1:
            return x[:fixed_size]
        fs = x

        # padding
        sx = stride * fixed_size - x.shape[0]
        if sx > 0:
            fs = np.pad(fs, (0, sx), mode='edge')
        
        # conv1d 
        fc = fs.reshape([1, fs.shape[0], 1, 1])
        kernel = np.ones(shape=[stride, 1, 1, 1]) / (stride)
        
        fc = conv2d(fc, kernel, 'SAME', (stride, 1))
        fc = np.squeeze(fc)
        return fc


    @classmethod
    def goes_conv2d_2(self, x, fixed_size):
        stride_x, stride_y = round(x.shape[0] / fixed_size[0]), round(x.shape[1] / fixed_size[1])
        if stride_x <= 1 and stride_y <= 1:
            return x[:fixed_size[0], :fixed_size[1]]
        fs = x

        # padding
        sx = stride_x * fixed_size[0] - x.shape[0]
        if sx > 0:
            fs = np.pad(fs, ((0, sx), (0, 0)), mode='edge')
        else:
            fs = fs[:stride_x * fixed_size[0], :]

        sy = stride_y * fixed_size[1] - x.shape[1]
        if sy > 0:
            fs = np.pad(fs, ((0, 0), (0, sy)), mode='edge')
        else:
            fs = fs[:, :stride_y * fixed_size[1]]
        
        # conv2d
        fc = fs.reshape([1, fs.shape[0], fs.shape[1], 1])
        kernel = np.ones(shape=[stride_x, stride_y, 1, 1]) / (stride_x * stride_y)
        
        fc = conv2d(fc, kernel, 'SAME', (stride_x, stride_y))
        fc = np.squeeze(fc)
        return fc


    @classmethod
    def scale_to_fixed_size(self, x, fixed_size):
        nx = []

        if isinstance(fixed_size, list) == True or isinstance(fixed_size, tuple) == True:
            if len(fixed_size) == 2:
                for i in x:
                    # nx.append( self.goes_conv2d(i, fixed_size) )
                    nx.append( self.goes_conv2d_2(i, fixed_size) )

        elif isinstance(fixed_size, int):
            for i in x:
                # nx.append( self.goes_conv1d(i, fixed_size) )
                nx.append( self.goes_conv1d_2(i, fixed_size) )

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


# 针对单一数据的统一size操作
# radiance, 经纬度和缩放到的大小(如果None则选择最小的)
def embedded_scale_method(rad_list, long_list=None, lati_list=None, size=None):
    if size is None:
        min_size = np.asarray(a=[9999, 9999])
        for rad in rad_list:
            rad_size = rad.shape
            min_size = np.where( min_size < rad_size, min_size, rad_size )
        min_size = list(min_size)
    else:
        min_size = size
    
    # rad part
    nrad = FixedScale.scale_to_fixed_size(rad_list, min_size)

    # long & lati part
    lati_min_size = int(min_size[0])
    long_min_size = int(min_size[1])
    
    if long_list is not None:
        ms_long = long_list[0]
        for lon in long_list:
            if(ms_long.shape[0] <= lon.shape[0]):
                ms_long = lon
        # nlong = FixedScale.scale_to_fixed_size(ms_long, long_min_size)
        nlong = FixedScale.goes_conv1d_2(ms_long, long_min_size)
    else:
        nlong = None

    if lati_list is not None:
        ms_lati = lati_list[0]
        for lat in lati_list:
            if(ms_lati.shape[0] <= lat.shape[0]):
                ms_lati = lat
        # nlati = FixedScale.scale_to_fixed_size(ms_lati, lati_min_size)
        nlati = FixedScale.goes_conv1d_2(ms_lati, lati_min_size)
    else:
        nlati = None

    return nrad, nlong, nlati


if __name__ == "__main__":
    # root_path = "./DataSet/Data/"
    # save_path = "./DataSet/ScaledData32/"

    # if os.path.exists(save_path) ==False:
    #     os.mkdir(save_path)
    # goes16_5channels_scale_dir(root_path, save_path, HurricaneExtraction.read_extraction_data)


    # root_path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\NpyData\\"
    # save_path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ScaledData32\\"

    # if os.path.exists(save_path) ==False:
    #     os.mkdir(save_path)
    # goes16_5channels_scale_dir(root_path, save_path, HurricaneExtraction.read_extraction_data)

    tmp = np.arange(start=0, stop=100)
    tmp = tmp.reshape([10, 10])
    print(tmp)
    print()

    # tmp3 = FixedScale.goes_conv2d_2(tmp, (3, 10))

    tmp2 = FixedScale.scale_to_fixed_size([tmp], (6, 10))
    print(tmp2)

    # tmp = np.arange(start=0, stop=10)
    # print(tmp)

    # tmp2 = FixedScale.scale_to_fixed_size([tmp], 10)
    # print(tmp2)
    