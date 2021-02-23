import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())  

from DataSet.extract import HurricaneExtraction, HurricaneExtractionRadM

nc_data_path = ""
best_track_file = ""
save_path_1 = ""

# 如果是RadM则用HurricaneExtractionRadM, 否则使用HurricaneExtraction
he = HurricaneExtractionRadM(nc_data_path, best_track_file, save_path_1, select_date=None)
he.hurricane_extraction()


# to be continued...

