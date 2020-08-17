import numpy as np
import os
import random

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

import DataSet.normalize as DN

# .\\IRMA\\Visible\\2017253
class HurricaneGenerator(object):
    def __init__(self):
        pass
    

    @classmethod
    def directory_downstream(self, root_path, wbl={}):
        if os.path.isdir(root_path) == False:
            return []
        
        white_list, black_list = None, None

        if 'white_list'in wbl:
            white_list = wbl['white_list']
        if 'black_list'in wbl:
            black_list = wbl['black_list']
        
        if white_list is not None and black_list is not None:
            print("黑名单-白名单最多只能存在一种")
            return []

        dirs = os.listdir(root_path)
        random.shuffle(dirs)
        sift = []

        for di in dirs:
            di_path = os.path.join(root_path, di)
            if os.path.isdir(di_path) == False:
                continue
            if black_list is not None:
                if di in black_list:
                    continue
            if white_list is not None:
                if di not in white_list:
                    continue
            sift.append(di_path)
        
        return sift


    @classmethod
    def leaf_directory_generator(self, root_path, wbl=[]):
        path = [root_path]

        for li in wbl:
            new_path = []
            for pth in path:
                dirs = HurricaneGenerator.directory_downstream(pth, li)
                new_path = new_path + dirs
            path.clear()
            path = new_path
        
        return path


    @classmethod
    def one_dircetory_generator(self, data_path, batch_size=1, read_data_func=None):
        datas = os.listdir(data_path)
        random.shuffle(datas)

        x = []
        for data in datas:
            dp = os.path.join(data_path, data)
            x.append(read_data_func(dp))
            if len(x) == batch_size:
                yield(np.array(x))


def name_visibility_date_dir_generator(root_path, read_data_func=None, batch_size=1, wbl=[{}, {}, {}]):
    def read_npy_hurricane_data(file_path):
        return np.load(file_path)

    if read_data_func is None:
        read_data_func = read_npy_hurricane_data
    
    leaf_directory = HurricaneGenerator.leaf_directory_generator(root_path, wbl)

    while True:
        random.shuffle(leaf_directory)
        for ld in leaf_directory:
            odg = HurricaneGenerator.one_dircetory_generator(ld, batch_size, read_data_func)
            while True:
                try:
                    hdg = next(odg)
                    hdg = DN.Quantization.convert_unsigned_to_float(hdg)      # 这里转化为浮点数
                    #hdg = HurricaneExtraction.normalize_using_physics(hdg)
                    yield(hdg)
                except StopIteration:
                    break


def name_visibility_date_dir_data_counts(root_path, black_list=None):
    final_dirs = []

    medi_dirs = [root_path]
    ptr = 0

    while ptr < len(medi_dirs):
        cur_path = medi_dirs[ptr]
        dirs = os.listdir(cur_path)
        flag  = 0
        for di in dirs:
            path = os.path.join(cur_path, di)
            if os.path.isdir(path) == False:
                continue
            if black_list is not None:
                if di not in black_list:
                    medi_dirs.append(path)
            else:
                medi_dirs.append(path)
            flag = 1
        if flag == 0:
            final_dirs.append(cur_path)
        ptr = ptr + 1
    
    count = 0
    for fd in final_dirs:
        count = count + len(os.listdir(fd))
    
    return count



if __name__ == "__main__":
    pass


