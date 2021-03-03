import numpy as np

# axis = 'x' / 'y' ： 上下翻转/水平翻转
def flip(data, axis='x'):
    if isinstance(data, np.ndarray) == False:
        print("不支持非np.ndarray类型")
        return data

    if data.ndim != 3 and data.ndim != 4:
        print("不支持2,3,4之外的维度")
        return data

    if data.ndim == 3:
        ndata = np.expand_dims(data, axis=0)
    else:
        ndata = data
    
    if axis == 'x':
        ndata = np.flip(ndata, axis=1)
    if axis == 'y':
        ndata = np.flip(ndata, axis=2)
    
    if data.ndim == 3:
        ndata = ndata[0]
    return ndata

    

# k = 1/-1 : 逆时针旋转90度,顺时针旋转90度
def rotation(data, k=1):
    if isinstance(data, np.ndarray) == False:
        print("不支持非np.ndarray类型")
        return data
    
    if data.ndim != 3 and data.ndim != 4:
        print("不支持2,3,4之外的维度")
        return data
    
    if data.ndim == 3:
        return np.rot90(data, k=k, axes=(0, 1))
    
    if data.ndim == 4:
        return np.rot90(data, k=k, axes=(1, 2)) 



def rotation90(data):
    return rotation(data, k=1)

def mirror_flip(data):
    return flip(data, axis='y')

def vertical_flip(data):
    return flip(data, axis='x')



def rotation90_in_list(data):
    for d in data:
        d = rotation90(d)
    return data

def mirror_flip_in_list(data):
    for d in data:
        d = mirror_flip(d)
    return data

def vertical_flip_in_list(data):
    for d in data:
        d = vertical_flip(d)
    return data



if __name__ == "__main__":
    
    import os
    import sys
    from PIL import Image

    sys.path.append(os.getcwd())
    from DataSet.quantization import Quantization
    from DataSet.normalize import Normalize

    data_path = 'D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ScaledData32\\IRMA\\Visible\\2017253\\20172531622.npy'

    data = np.load(data_path)
    data = Quantization.convert_unsigned_to_float(data)
    data = Normalize.normalize_using_physics(data)

    height = data.shape[0]
    width = data.shape[1]
    channel = data.shape[2]

    x = []
    x.append(data)
    x.append(flip(data, axis='x'))
    x.append(flip(data, axis='y'))
    x.append(rotation(data, k=1))
    x.append(rotation(data, k=2))
    x.append(rotation90(data))

    figure = np.zeros( (height * len(x), width * channel, 1) )

    for i in range(len(x)):
        for j in range(channel):
            figure[i*height : (i+1)*height, j*width : (j+1)*width] = x[i][..., j : (j+1)]

    figure = np.squeeze(figure)
    img = Image.fromarray(figure)

    save_name = str(1) + ".tiff"
    img.save(os.path.join("D:\\", save_name))
