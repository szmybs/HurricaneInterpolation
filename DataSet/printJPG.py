import numpy as np
import os
import sys
import glob
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from DataSet.quantization import Quantization
from DataSet.hurricane_generator import HurricaneGenerator, hurricane_load
from DataSet.normalize import Normalize

root_path = "./DataSet/ScaledData256-RadM/"
save_path = "./DataSet/Data-JPG/"


def printJPG():
    leaf_directory = HurricaneGenerator.leaf_directory_generator(root_path=root_path, wbl=[ {}, {'white_list':['Visible']}, {} ])
    for ld in leaf_directory:
        nld = ld.replace("ScaledData256-RadM", "Data-JPG")
        if os.path.exists(nld) == False:
            os.makedirs(nld)

        npy_files = glob.glob(os.path.join(ld, "*.npy"))

        for nf in npy_files:
            data = hurricane_load(nf)
            data = data[..., 0]

            figure = np.squeeze(data)
            figure =  figure * 255         # 将0-1变为0-255，似乎很多软件只能打开0-255式的
            
            img = Image.fromarray(np.int32(figure), mode='I')
            img = img.convert("RGB")

            sp = os.path.join(nld, (os.path.basename(nf)).replace("npy", "jpg"))
            img.save(sp)


# 打印成jpg图像后手动删除不太行的，然后根据还留存的jpg图像保留台风数据
def handy_filter_hurricane_data(npy_directory):
    npy_leaf = HurricaneGenerator.leaf_directory_generator(root_path=npy_directory, wbl=[ {}, {'white_list':['Visible']}, {} ])

    for nl in npy_leaf:
        jl = nl.replace("ScaledData256-RadM", "Data-JPG")

        npy_files = glob.glob(os.path.join(nl, "*.npy"))
        for nf in npy_files:
            jf = os.path.join(jl, (os.path.basename(nf)).replace("npy", "jpg") )
            if os.path.exists(jf) == False:
                print(nf)
                os.remove(nf)
                if not os.listdir( os.path.dirname(nf) ):
                    os.rmdir( os.path.dirname(nf) )

handy_filter_hurricane_data(root_path)



