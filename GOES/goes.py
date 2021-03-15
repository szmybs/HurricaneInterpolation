from netCDF4 import Dataset
import numpy as np
import os
import math
import datetime

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.time_format import convert_julian_to_datetime
from DataSet.scale import embedded_scale_method


#OR_ABI-L1b-RadC-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
#OR_ABI-L1b-RadM1-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
def goes_file_name_seperation(file_name):
    file_name = os.path.split(file_name)[1]
    seg = file_name.split('_')

    sensor = seg[1][:12]
    channel = seg[1][-5:]

    year = seg[3][1:5]
    jd = seg[3][5:8]
    hour = seg[3][8:10]
    minute = seg[3][10:12]
    return (sensor, channel, year, jd, hour, minute, convert_julian_to_datetime(year+jd+hour+minute))



class netCDF_Head(object):
    def __init__(self, g16nc):
        self.y = g16nc.variables['y'][:]
        self.x = g16nc.variables['x'][:]

        # 卫星星下点的经度坐标
        self.geospatial_lon_nadir = g16nc.variables['geospatial_lat_lon_extent'].geospatial_lon_nadir

        # 卫星云图边界，经过缩放后可能改变
        self.y_image_bound = g16nc.variables['y_image_bounds'][:]
        self.x_image_bound = g16nc.variables['x_image_bounds'][:]

        # using to convert between reflectance factors and radiances(W/ m2•sr• µm)
        self.kappa0 = g16nc.variables['kappa0'][:]
    

    # 看起来由于尾数的误差导致一直返回false，弃用
    def check(self, g16nc):
        if self.y != g16nc.variables['y'][:]:
            return False
        
        if self.x != g16nc.variables['x'][:]:
            return False
        
        if self.y_image_bound != g16nc.variables['y_image_bounds'][:]:
            return False
        
        if self.x_image_bound != g16nc.variables['x_image_bounds'][:]:
            return False
        return True



class GOES_netCDF(object):
    def __init__(self, file_names, **kwargs):
        if 'Rad' in kwargs and 'Date' in kwargs and 'Head' in kwargs:
            self.Rad = kwargs['Rad']
            self.date = kwargs['Date']
            self.data_head = kwargs['Head']
        else:
            self.Rad = []
            self.wave_length = []
            for fn in file_names:
                g16nc = Dataset(fn)
                self.Rad.append(g16nc.variables['Rad'][:])
                self.wave_length.append(g16nc.variables['band_wavelength'][:])

                if hasattr(self, 'data_head') == False:
                    self.data_head = netCDF_Head(g16nc)

                flagT = self._get_datetime(fn)

                g16nc.close()

                if flagT == False:
                    self.Rad.clear()
                    delattr(self, 'data_head')
                    delattr(self, 'date')
                    print("各波段数据日期不同！")
                    break
            
            self._scale()
    

    def _get_datetime(self, file_name):
        if hasattr(self, 'date') == False:
            self.date = goes_file_name_seperation(file_name)[-1]
        else:
            if self.date != goes_file_name_seperation(file_name)[-1]:
                return False
        return True


    def _scale(self):
        r, y, x = embedded_scale_method(self.Rad, [self.y], [self.x])
        self.Rad = r
        self.data_head.y = y
        self.data_head.x = x


    def add_longitude_table(self, longitude_table):
        self.longitude_table = longitude_table
    
    def add_latitude_table(self, latitude_table):
        self.latitude_table = latitude_table
    

    @property
    def geospatial_lon_nadir(self):
        return self.data_head.geospatial_lon_nadir
    
    @property
    def shape(self):
        return (self.data_head.y.shape[0], self.data_head.x.shape[0])

    @property
    def y(self):
        return self.data_head.y

    @property
    def x(self):
        return self.data_head.x

    @property
    def y_image_bound(self):
        return self.data_head.y_image_bound

    @property
    def x_image_bound(self):
        return self.data_head.x_image_bound

    @property
    def kappa0(self):
        return self.data_head.kappa0



if __name__ == "__main__":
    pass

    # y = [0.095340, 0.095340, 0.095340]
    # x = [-0.024052, -0.024052, -0.024052, -0.024052, -0.024052]
    # y = [1, 2, 3]
    # x = [4, 5, 6, 7, 8]
    #tmp = GOES.navigating_from_elevation_scanning_angle_to_geodetic(0.095340, -0.024052, -1.308996939)   
    #tmp2 = GOES.navigating_from_elevation_scanning_angle_to_geodetic2(np.asarray(y), np.asarray(x), -1.308996939)
    # GOES.compute_latitued_longitued_per_ABI_grid(np.asarray(y), np.asarray(x), -1.308996939)

    # file_name = "OR_ABI-L1b-RadC-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc"
    # #file_name = "OR_ABI-L1b-RadM1-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc"
    # goes_file_name_seperation(file_name)

