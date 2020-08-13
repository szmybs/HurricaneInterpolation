import numpy as np
import os
import random

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from DataSet.extract import HurricaneExtraction

# .\\IRMA\\Visible\\2017253
class HurricaneGenerator(object):
    def __init__(self):
        pass
    
    @classmethod
    def directory_downstream(self, root_path, white_list=None, black_list=None):
        if os.path.isdir(root_path) == False:
            return []

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
    def one_dircetory_generator(self, data_path, batch_size=1, read_data_func=None):
        datas = os.listdir(data_path)
        random.shuffle(datas)

        x = []
        for data in datas:
            dp = os.path.join(data_path, data)
            x.append(read_data_func(dp))
            if len(x) == batch_size:
                yield(np.array(x))



def name_visibility_date_dir_generator(root_path, batch_size=1, read_data_func=None, **kwarg):
    def read_npy_hurricane_data(file_path):
        return np.load(file_path)

    if read_data_func is None:
        read_data_func = read_npy_hurricane_data
    
    while True:

        if 'hurricane_name_white_list' in kwarg:
            name_dirs = HurricaneGenerator.directory_downstream(root_path, white_list=kwarg['hurricane_name_white_list'])
        elif 'hurricane_name_black_list' in kwarg:
            name_dirs = HurricaneGenerator.directory_downstream(root_path, black_list=kwarg['hurricane_name_black_list'])
        else:
            name_dirs = HurricaneGenerator.directory_downstream(root_path)

        for name_dir in name_dirs:

            if 'Visible' in kwarg:
                visibility_dirs = HurricaneGenerator.directory_downstream(name_dir, white_list=kwarg['Visible'])
            elif 'Invisible' in kwarg:
                visibility_dirs = HurricaneGenerator.directory_downstream(name_dir, white_list=kwarg['Invisible'])
            else:
                visibility_dirs = HurricaneGenerator.directory_downstream(name_dir)

            for visibility_dir in visibility_dirs:

                date_dirs = HurricaneGenerator.directory_downstream(visibility_dir)
                for date_dir in date_dirs:

                    odg = HurricaneGenerator.one_dircetory_generator(date_dir, batch_size, read_data_func)  #使用读取npy的内部函数
                    while True:
                        try:
                            hdg = next(odg)
                            hdg = HurricaneExtraction.convert_unsigned_to_float(hdg)  #在这里normlize
                            hdg = HurricaneExtraction.normalize_using_physics(hdg)
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
    # black_list = ['Visible']
    # dirs = name_visibility_date_dir_data_counts(".\\Data\\NpyData\\", black_list=None)
    # print(dirs)

    root_path = "./DataSet/ScaledData/"
    nvdd = name_visibility_date_dir_generator(root_path, 8)

    for i in range(10):
        tmp = next(nvdd)
        print(tmp)



