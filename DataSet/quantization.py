import numpy as np

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
 