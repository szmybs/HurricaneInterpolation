from netCDF4 import Dataset
import numpy as np
import os
import math
import datetime

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.time_format import convert_julian_to_datetime 


#OR_ABI-L1b-RadC-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
#OR_ABI-L1b-RadM1-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
def goes_file_name_seperation(file_name):
    seg = file_name.split('_')

    sensor = seg[1][:12]
    channel = seg[1][-5:]

    year = seg[3][1:5]
    jd = seg[3][5:8]
    hour = seg[3][8:10]
    minute = seg[3][10:12]
    return (sensor, channel, year, jd, hour, minute, convert_julian_to_datetime(year+jd+hour+minute))



class GOES(object):
    @classmethod
    def navigating_from_geodetic_to_elevation_scanning_angle(self, latitude, longitude, lamb0):
        phi = math.radians(latitude)
        lamb = math.radians(longitude)
        lamb0 = math.radians(lamb0)

        req = 6378137  #goes_imagery_projection:semi_major_axis /meters
        f = 298.257222096  #goes_imagery_projection:inverse_flattening
        rpol = 6356752.31414 #goes_imagery_projection:semi_minor_axis /meters
        e = 0.0818191910435
        ggh = 35786023 #goes_imagery_projection:perspective_point_height
        H = ggh + req

        phiC = math.atan( rpol*rpol*math.tan(phi) / (req*req) )
        rC = rpol / math.sqrt( 1 - math.pow(e*math.cos(phiC), 2) )

        sx = H - rC * math.cos(phiC) * math.cos(lamb-lamb0)
        sy = -1 * rC * math.cos(phiC) * math.sin(lamb-lamb0)
        sz = rC * math.sin(phiC)

        visible = H * (H-sx) - sy*sy - math.pow(req,2) * math.pow(sz,2) / math.pow(rpol,2)
        if visible < 0:
            return (-1, -1)
        
        y = math.atan2(sz, sx)
        x = math.asin( -1*sy / math.sqrt( sx*sx + sy*sy + sz*sz ) )
        return (y, x)


    @classmethod
    def navigating_from_elevation_scanning_angle_to_geodetic(self, y, x, lamb0):
        lamb0 = math.radians(lamb0)

        req = 6378137  #goes_imagery_projection:semi_major_axis /meters
        f = 298.257222096  #goes_imagery_projection:inverse_flattening
        rpol = 6356752.31414 #goes_imagery_projection:semi_minor_axis /meters
        e = 0.0818191910435
        ggh = 35786023 #goes_imagery_projection:perspective_point_height
        H = ggh + req

        sinx = np.sin(x)
        cosx = np.cos(x)
        siny = np.sin(y)
        cosy = np.cos(y)

        a = np.square(sinx) + np.square(cosx)*(np.square(cosy) + ((req**2 / rpol**2)*np.square(siny)))
        b = -2 * H * np.multiply(cosx, cosy)
        c = H**2 - req**2
        
        rs = np.divide( np.subtract(-b, np.sqrt(np.subtract(np.square(b), 4*np.multiply(a,c)))), 2*a )
        sx = np.multiply(np.multiply(rs, cosx), cosy)
        sy = np.multiply(-rs, sinx) 
        sz = np.multiply(np.multiply(rs, cosx), siny)

        ltmp = (req**2 / rpol**2) * (np.divide(sz, np.sqrt( np.square(H-sx) + np.square(sy) )))
        latitude = np.arctan(ltmp)
        longitude = lamb0 - np.arctan(np.divide(sy, H-sx))

        # 这里应该是 -180 ~ 180
        return np.degrees(latitude), np.degrees(longitude)
        # return latitude, longitude


    @classmethod
    def compute_latitude_longitude_table(self, y, x, lamb0):
        new_shape = (y.shape[0], x.shape[0])
        zeros = np.zeros(new_shape)

        y = np.expand_dims(y, axis=1)
        y = y + zeros

        x = np.expand_dims(x, axis=0)
        x = x + zeros

        latitude, longitude = self.navigating_from_elevation_scanning_angle_to_geodetic(y, x, lamb0)
        
        return latitude, longitude



class netCDF_Head(object):
    def __init__(self, g16nc):
        self.y = g16nc.variables['y'][:]
        self.x = g16nc.variables['x'][:]

        # 卫星星下点的经度坐标
        self.geospatial_lon_nadir = g16nc.variables['geospatial_lat_lon_extent'].geospatial_lon_nadir
        self.y_image_bound = g16nc.variables['y_image_bounds'][:]
        self.x_image_bound = g16nc.variables['x_image_bounds'][:]

        # using to convert between reflectance factors and radiances(W/ m2•sr• µm)
        self.kappa0 = g16nc.variables['kappa0'][:]
    
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
            for fn in file_names:
                g16nc = Dataset(fn)
                self.Rad.append(g16nc.variables['Rad'][:])

                flagD = self._get_data_head(g16nc)
                flagT = self._get_datetime(fn)

                g16nc.close()

                if flagD == False or flagT == False:
                    self.Rad.clear()
                    delattr(self, 'data_head')
                    delattr(self, 'date')
                    print("数据一致性检验错误")
                    break
            self.Rad = np.ndarray(self.Rad)


    def _get_data_head(self, g16nc):
        if hasattr(self, 'data_head') == False:
            self.data_head = netCDF_Head(g16nc)
        else:
            return self.data_head.check(g16nc)
        return True
    
    def _get_datetime(self, file_name):
        if hasattr(self, 'date') == False:
            self.date = goes_file_name_seperation(file_name)[-1]
        else:
            if self.date != goes_file_name_seperation(file_name)[-1]:
                return False
        return True

    def add_longitude_table(self, longitude_table):
        self.longitude_table = longitude_table
    
    def add_latitude_table(self, latitude_table):
        self.latitude_table = latitude_table



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

