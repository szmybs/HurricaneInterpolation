import numpy as np
import os
import glob

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from DataSet.quantization import Quantization
# from DataSet.hurricane_generator import HurricaneGenerator
from DataSet import hurricane_generator as G


class Normalize(object):
    def __init__(self, data_path, gaussian_path=None, max_min_path=None):
        self.data_path = data_path
        self.gaussian_path = gaussian_path
        self.max_min_path = max_min_path
    

    '''
    M3C01_Range = [-25.937, 804.036]
    M3C07_Range = [-0.0376, 25.590]
    M3C09_Range = [-0.824, 45.291]
    M3C14_Range = [-1.719, 200.902]
    M3C15_Range = [-1.756, 214.301]
    '''
    @classmethod
    def normalize_using_physics(self, data):
        Range = [[-25.937, 804.036],
                 [-0.0376, 25.590],
                 [-0.824, 45.291],
                 [-1.719, 200.902],
                 [-1.756, 214.301]]

        if isinstance(data, list):
            norm_data = []
            for i in range(len(data)):
                tmp = (data[i] - Range[i][0]) / (Range[i][1] - Range[i][0])
                norm_data.append(tmp)
            return norm_data

        if isinstance(data, np.ndarray):
            min_range = np.array( [ Range[0][0], Range[1][0], Range[2][0], Range[3][0], Range[4][0] ] )
            max_range = np.array( [ Range[0][1], Range[1][1], Range[2][1], Range[3][1], Range[4][1] ] )

            norm_data = np.divide( np.subtract(data, min_range), np.subtract(max_range, min_range) )
            return norm_data

    @classmethod
    def undo_normalize_using_physics(self, data):
        Range = [[-25.937, 804.036],
                 [-0.0376, 25.590],
                 [-0.824, 45.291],
                 [-1.719, 200.902],
                 [-1.756, 214.301]]
        
        if isinstance(data, list):
            unnorm_data = []
            for i in range(len(data)):
                tmp = data[i] * (Range[i][1] - Range[i][0]) + Range[i][0]
                unnorm_data.append(tmp)
            return unnorm_data

        if isinstance(data, np.ndarray):
            min_range = np.array( [ Range[0][0], Range[1][0], Range[2][0], Range[3][0], Range[4][0] ] )
            max_range = np.array( [ Range[0][1], Range[1][1], Range[2][1], Range[3][1], Range[4][1] ] )

            unnorm_data = np.add( np.multiply(data, np.subtract(max_range, min_range)), min_range )
            return unnorm_data

    def _get_files(self):
        lds = G.HurricaneGenerator.leaf_directory_generator(self.data_path, wbl=[{}, {'white_list':['Visible']}, {}])
        files = []
        for ld in lds:
            f = glob.glob(pathname=os.path.join(ld, '*.npy'))
            files = files + f
        return files

    def normalize_using_std_gaussian(self, data):
        if os.path.exists(self.gaussian_path) == False:
            files = self._get_files()            
            means = []
            for fi in files:
                fptr = np.load(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                means.append( np.mean(fptr, axis=(0, 1)) )
            means = np.array(means)
            means = np.mean(means, axis=0)

            var = []
            for fi in files:
                fptr = np.load(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                fptr = self.normalize_using_physics(fptr)
                subm = np.subtract(fptr, means)
                mult = np.multiply(subm, subm)
                var.append( np.mean(mult, axis=(0, 1)) )
            var = np.array(var)
            var = np.mean(var, axis=0)
            std = np.power(var, 0.5)            
            np.savez(file=self.gaussian_path, mean=means, std=std)

        if hasattr(self, 'mean') == False or hasattr(self, 'std') == False:
            dptr = np.load(self.gaussian_path)
            self.mean = dptr['mean']
            self.std = dptr['std']
        
        return  np.divide( np.subtract(data, self.mean), self.std )

    def undo_normalize_using_std_gaussian(self, data):
        try:
            if hasattr(self, 'mean') == False or hasattr(self, 'std') == False:
                dptr = np.load(self.gaussian_path)
                self.mean = dptr['mean']
                self.std = dptr['std']
            return np.add( np.multiply(data, self.std), self.mean )
        except OSError:
            print("需求文件打不开")
            return data


    def normalize_using_max_min(self, data, mode=0):
        if os.path.exists(self.max_min_path) == False:
            files = self._get_files()      
            MAX, MIN = [], []
            for fi in files:
                fptr = np.load(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                fptr = self.normalize_using_physics(fptr)
                MAX.append( np.amax(fptr, axis=(0, 1)) )
                MIN.append( np.amin(fptr, axis=(0, 1)) )
            MAX = np.array(MAX)
            MIN = np.array(MIN)
            MAX = np.amax(MAX, axis=0)
            MIN = np.amin(MIN, axis=0)
         
            np.savez(file=self.max_min_path, max=MAX, min=MIN)

        if hasattr(self, 'max') == False and hasattr(self, 'min') == False:
            dptr = np.load(self.max_min_path)
            self.max = dptr['max']
            self.min = dptr['min']
        
        if mode == 0:
            return  np.divide( np.subtract(data, self.min), np.subtract(self.max, self.min) )
        return np.divide( np.subtract(data, self.min), np.subtract(self.max, self.min)/2 ) - 1

    def undo_normalize_using_max_min(self, data, mode=0):
        try:
            if hasattr(self, 'max') == False or hasattr(self, 'min') == False:
                dptr = np.load(self.max_min_path)
                self.max = dptr['max']
                self.min = dptr['min']
            if mode == 0:
                return np.add( np.multiply(data, np.subtract(self.max, self.min)), self.min )
            return np.add( np.multiply( (data + 1),  np.subtract(self.max, self.min)/2 ), self.min )
        except OSError:
            print("需求文件打不开")
            return data


if __name__ == "__main__":
    pass