import numpy as np
import os

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.goes import GOES_netCDF
from Modeling.zenith_azimuth import solar_satellite_zenith_azimuth_angle


class HurricaneModel(object):
    def __init__(self, files_name, **kwargs):
        if isinstance(files_name, list):
            self.g16nc = GOES_netCDF(files_name)
        else:
            rad = np.load(files_name)
            head, date = kwargs['Head'], kwargs['Date']
            self.g16nc = GOES_netCDF(Rad=rad, Head=head, Date=date)

        # Band 7 ~ 16 : 7,9,14,15
        self.temperature = self.conversion_radiances_to_brightness_temperature(self.g16nc.Rad[1:])
        self.convsersion_bands_nums()

        self.solar_zenith_angle, self.solar_azimuth_angle, self.satellite_zenith_angle, self.satellite_azimuth_angle = solar_satellite_zenith_azimuth_angle(self.g16nc)
        # Band 1 ~ 6 : only 1
        self.refVIS = self.conversion_radiances_to_reflectance(self.g16nc.Rad[:1], self.g16nc.data_head.kappa0, self.solar_zenith_angle)


    # 无云:0  水云:1  冰云:2  卷云:3
    def cloud_classification(self):
        ice_cloud 


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


    def convsersion_bands_nums(self):
        # VIS-M3C01  IR1-M3C14  IR2-M3C15  WV-M3C09  MWIR-M3C07
        # M3C01-VIS  M3C07-MWIR  M3C09-WV  M3C14-IR1  M3C15-IR2
        self.tempIR1 = self.temperature[-2]
        self.tempIR2 = self.temperature[-1]
        self.tempWV = self.temperature[-3]
        self.tempMWIR = self.temperature[-4]
        pass


if __name__ == "__main__":
    pass