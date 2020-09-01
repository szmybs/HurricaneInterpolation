import numpy as np
import os

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.goes import GOES_netCDF
from Modeling.zenith_azimuth import solar_satellite_zenith_azimuth_angle


class HurricaneModel(object):
    def __init__(self, files_name):
        if isinstance(files_name, list):
            self.rad = GOES_netCDF(files_name)
        else:
            self.rad = np.load(files_name)

        self.temperature = self.conversion_radiances_to_brightness_temperature(self.rad[1:])


    def cloud_classification(self):



    @classmethod
    def conversion_radiances_to_brightness_temperature(self, rad):
        # 该方法针对Band7 ~ 16

        planck_fk1 = [2.02263e+05, 3.58283e+03, 8.51022e+03, 6.45462e+03]
        planck_fk2 = [3.69819e+03, 2.07695e+03, 1.28627e+03, 1.17303e+03]
        planck_bc1 = [0.43361, 0.34427, 0.22516, 0.21702]
        planck_bc2 = [0.99939, 0.99918, 0.99920, 0.99916]

        if isinstance(rad, list):
            btemp = []
            for i in range(len(rad)):
                bt = ((planck_fk2[i] /  np.log( (planck_fk1[i] / rad[i]) + 1)) - planck_bc1[i]) / planck_bc2[i]
                btemp.append(bt)
            return btemp

        if isinstance(rad, np.ndarray):
            p1 = np.divide( np.array(planck_fk2), np.log( np.divide(np.array(planck_fk1), rad) +1 ) )
            p2 = np.divide( np.array(planck_bc1), np.array(planck_bc2) )
            return np.subtract(p1, p2)
    
    
    @classmethod
    def conversion_radiances_to_reflectance(self, rad, kappa0, solar_zenith_angle):
        # 该方法针对 Band1 ~ Band6
        return np.divide( (rad * kappa0), np.cos(solar_zenith_angle) )


    @classmethod
    def convsersion_bands_nums(self, rad):
        # VIS-M3C01  IR1-M3C14  IR2-M3C15  WV-M3C09  MWIR-M3C07
        # M3C01-VIS  M3C07-MWIR  M3C09-WV  M3C14-IR1  M3C15-IR2
        pass