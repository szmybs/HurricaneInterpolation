import numpy as np
import os
import glob

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from DataSet.hurricane_generator import HurricaneGenerator


class Quantization(object):
    def __init__(self):
        pass

    @classmethod
    def convert_float_to_unsigned(self, float_data):
        scale_factor = (0.812106364, 0.001564351, 0.022539101,
                        0.049492208, 0.052774108)
        add_offset = (-25.93664701, -0.03760000, -0.82360000,
                      -1.71870000, -1.75580000)
        
        if isinstance(float_data, list):
            unsigned_data = []
            for i in range(len(float_data)):
                tmp = (float_data[i] - add_offset[i]) / scale_factor[i]
                tmp = np.around(tmp, 0)
                tmp = tmp.astype(np.uint16)
                unsigned_data.append(tmp)
            return unsigned_data
        
        if isinstance(float_data, np.ndarray):
            sf = np.array(scale_factor)
            ao = np.array(add_offset)

            unsigned_data = np.divide( np.subtract(float_data - ao), sf )
            unsigned_data = (np.around(unsigned_data, 0)).astype(np.uint16)
            return unsigned_data 

    @classmethod
    def convert_unsigned_to_float(self, unsigned_data):
        scale_factor = (0.812106364, 0.001564351, 0.022539101,
                        0.049492208, 0.052774108)
        add_offset = (-25.93664701, -0.03760000, -0.82360000,
                      -1.71870000, -1.75580000)
        
        if isinstance(unsigned_data, list):
            float_data = []
            for i in range(len(unsigned_data)):
                tmp = unsigned_data[i] * scale_factor[i] + add_offset[i]
                tmp = tmp.astype(np.float32)
                float_data.append(tmp)
            return float_data
        
        if isinstance(unsigned_data, np.ndarray):
            sf = np.array(scale_factor)
            ao = np.array(add_offset)

            float_data = np.add( np.multiply( unsigned_data, sf ), ao)
            float_data = float_data.astype(np.float32)
            return float_data
 

class Normalize(object):
    def __init__(self, data_path):
        self.data_path = data_path
    

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


    def normalize_using_std_gaussian(self, data, path):
        if os.path.exists(path) == False:
            lds = HurricaneGenerator.leaf_directory_generator(self.data_path, wbl=[{}, {'white_list':['Visible']}, {}])
            files = []
            for ld in lds:
                f = glob.glob(pathname=os.path.join(ld, '*.npy'))
                files = files + f
            
            means = []
            for fi in files:
                fptr = np.open(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                means.append( np.mean(fptr, axis=(0, 1)) )
            means = np.array(means)
            means = np.mean(means, axis=0)

            var = []
            for fi in files:
                fptr = np.load(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                subm = np.subtract(fptr, means)
                mult = np.multiply(subm, subm)
                var.append( np.mean(mult, axis=(0, 1)) )
            var = np.array(var)
            var = np.mean(var, axis=0)
            std = np.power(var, 0.5)            
            np.savez(file=path, mean=means, std=std)

        if hasattr(self, 'mean') == False or hasattr(self, 'std') == False:
            dptr = np.load(path)
            self.mean = dptr['mean']
            self.std = dptr['std']
        
        return  np.divide( np.subtract(data - self.mean), self.std )

    def undo_normalize_using_std_gaussian(self, data, path):
        try:
            if hasattr(self, 'mean') == False or hasattr(self, 'std') == False:
                dptr = np.load(path)
                self.mean = dptr['mean']
                self.std = dptr['std']
            return np.add( np.multiply(data, self.std), self.mean )
        except OSError:
            print("需求文件打不开")
            return data


    def normalize_using_max_min(self, data, path, mode=0):
        if os.path.exists(path) == False:
            lds = HurricaneGenerator.leaf_directory_generator(self.data_path, wbl=[{}, {'white_list':['Visible']}, {}])
            files = []
            for ld in lds:
                f = glob.glob(pathname=os.path.join(ld, '*.npy'))
                files = files + f
            
            MAX, MIN = [], []
            for fi in files:
                fptr = np.load(fi)
                fptr = Quantization.convert_unsigned_to_float(fptr)
                MAX.append( np.amax(fptr, axis=(0, 1)) )
                MIN.append( np.amin(fptr, axis=(0, 1)) )
            MAX = np.array(MAX)
            MIN = np.array(MIN)
            MAX = np.amax(MAX, axis=0)
            MIN = np.amin(MIN, axis=0)
         
            np.savez(file=path, max=MAX, min=MIN)

        if hasattr(self, 'max') == False and hasattr(self, 'min') == False:
            dptr = np.load(path)
            self.max = dptr['max']
            self.min = dptr['min']
        
        if mode == 0:
            return  np.divide( np.subtract(data - self.min), np.subtract(self.max - self.min) )
        return np.divide( np.subtract(data - self.min), np.subtract(self.max - self.min)/2 ) - 1

    def undo_normalize_using_max_min(self, data, path, mode=0):
        try:
            if hasattr(self, 'max') == False or hasattr(self, 'min') == False:
                dptr = np.load(path)
                self.mean = dptr['max']
                self.std = dptr['min']
            if mode == 0:
                return np.add( np.multiply(data, np.subtract(self.max, self.min)), self.min )
            return np.add( np.multiply( (data + 1),  np.subtract(self.max - self.min)/2 ), self.min )
        except OSError:
            print("需求文件打不开")
            return data


if __name__ == "__main__":
    data_path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ScaledData32\\"
    norm = Normalize(data_path)

    data = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ScaledData32\\IRMA\\Visible\\2017253\\20172531622.npy"
    data = np.load(data)

    norm.normalize_using_max_min(data, path=".\\DataSet\\max_min.npz")