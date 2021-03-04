import numpy as np
import os
import random

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from DataSet.quantization import Quantization
# from DataSet.normalize import Normalize
from DataSet import normalize as N

# .\\IRMA\\Visible\\2017253
class HurricaneGenerator(object):
    def __init__(self):
        pass
    
    @classmethod
    def directory_downstream(self, root_path, mode='random', wbl={}):
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
        if mode == 'random':
            random.shuffle(dirs)
        elif mode == 'sort':
            dirs.sort()
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
    def leaf_directory_generator(self, root_path, mode='random', wbl=[]):
        path = [root_path]

        for li in wbl:
            new_path = []
            for pth in path:
                dirs = HurricaneGenerator.directory_downstream(pth, mode, li)
                new_path = new_path + dirs
            path.clear()
            path = new_path
        
        return path


    @classmethod
    def one_dircetory_generator(self, data_path, batch_size=1, mode='random', read_data_func=None):
        datas = os.listdir(data_path)
        if mode == 'random':
            random.shuffle(datas)
        elif mode == 'sort':
            datas.sort()

        x = []
        for data in datas:
            dp = os.path.join(data_path, data)
            x.append(read_data_func(dp))
            if len(x) == batch_size:
                yield(np.array(x))
                x.clear()


# 还是在这里进行physics标准化. 不然总要加上这句太烦了
def hurricane_load(file_path):
    data = np.load(file_path)
    data = Quantization.convert_unsigned_to_float(data)
    # data = Normalize.normalize_using_physics(data)
    data = N.Normalize.normalize_using_physics(data)
    return data


def name_visibility_date_dir_generator(root_path, read_data_func=None, batch_size=1, wbl=[{}, {}, {}]):
    def read_npy_hurricane_data(file_path):
        return hurricane_load(file_path)

    if read_data_func is None:
        read_data_func = read_npy_hurricane_data
    
    leaf_directory = HurricaneGenerator.leaf_directory_generator(root_path=root_path, wbl=wbl)

    while True:
        random.shuffle(leaf_directory)
        for ld in leaf_directory:
            odg = HurricaneGenerator.one_dircetory_generator(ld, batch_size, 'random', read_data_func)
            while True:
                try:
                    hdg = next(odg)
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


# 在batch_size 不等于 1 的情况下存在问题
def name_visibility_date_dir_seq_generator(root_path, read_data_func=None, batch_size=1, length=3, wbl=[{}, {}, {}]):
    def read_npy_hurricane_data(file_path):
        return hurricane_load(file_path)

    if read_data_func is None:
        read_data_func = read_npy_hurricane_data
    
    leaf_directory = HurricaneGenerator.leaf_directory_generator(root_path=root_path, mode='sort', wbl=wbl)

    while True:
        random.shuffle(leaf_directory)
        for ld in leaf_directory:
            odg = HurricaneGenerator.one_dircetory_generator(data_path=ld, batch_size=batch_size, mode='sort', read_data_func=read_data_func)
                   
            seq = []
            while True:
                try:
                    while len(seq) < length:
                        hdg = next(odg)
                        seq.append(hdg)

                    yield(seq)
                    seq.clear()
                except StopIteration:
                    break


def name_visibility_date_dir_seq_generator_V2(root_path, read_data_func=None, batch_size=1, length=3, wbl=[{}, {}, {}]):
    def read_npy_hurricane_data(file_path):
        return hurricane_load(file_path)

    if read_data_func is None:
        read_data_func = read_npy_hurricane_data
    
    leaf_directory = HurricaneGenerator.leaf_directory_generator(root_path=root_path, mode='sort', wbl=wbl)

    while True:
        random.shuffle(leaf_directory)
        for ld in leaf_directory:
            # odg = HurricaneGenerator.one_dircetory_generator(data_path=ld, batch_size=batch_size, mode='sort', read_data_func=read_data_func)

            necessity_size = (batch_size - 1) + length
            odg = HurricaneGenerator.one_dircetory_generator(data_path=ld, batch_size=necessity_size, mode='sort', read_data_func=read_data_func)     
            seq = []
            while True:
                try:
                    hdg = next(odg)
                    
                    # while len(seq) < batch_size:
                    #     seq.append(hdg[len(seq):len(seq)+length])
                    # yield(seq)
                    while len(seq) < length:
                        seq.append(hdg[len(seq):len(seq)+length])
                    yield(seq)

                    seq.clear()
                except StopIteration:
                    break



if __name__ == "__main__":
    pass

