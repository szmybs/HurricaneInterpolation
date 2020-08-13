import numpy as np
import glob
import os

medi_dirs = ['./DataSet/ScaledData/']
ptr = 0

while ptr < len(medi_dirs):
    cur_path = medi_dirs[ptr]
    dirs = os.listdir(cur_path)
    for di in dirs:
        path = os.path.join(cur_path, di)
        if os.path.isdir(path) == False:
            continue
        medi_dirs.append(path)
    ptr = ptr + 1

std_size = 655488

for medi_dir in medi_dirs:
    npy_files = glob.glob(os.path.join(medi_dir, '*.npy'))
    for nf in npy_files:
        st = os.stat(nf)
        if st.st_size != std_size:
            print(nf)

print("OVER")
