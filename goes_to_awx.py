import numpy as np
import math
import os
import datetime
import struct
from PIL import Image
from netCDF4 import Dataset

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.goes import netCDF_Head
from GOES.goes_calculation import GOES
from DataSet.scale import embedded_scale_method
from Modeling.HurricaneModel import HurricaneModel



class FileNameProcess(object):

    @classmethod
    def time_format_convert(self, date, to_julian=True):
        if type(date) != 'str':
            date = str(date)

        leap_year = False
        year = int(date[:4])
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    leap_year = True
            else:
                leap_year = True

        if leap_year == False:
            md = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #2017
        else:
            md = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if to_julian == True:
            month = int(date[4:6])
            day = int(date[6:8])

            julian_day = day
            for i in range(month):
                julian_day += md[i]
            julian_day = '00' + str(julian_day)
            julian_day = julian_day[-3:]

            return (str(year)+julian_day, int(year), int(julian_day))
        else:
            julian_day = date[4:7]
            month = 0
            day = int(julian_day)
            for i in md:
                if day > i:
                    month = month + 1
                    day = day - i
                else:
                    break

            hour = 0
            minute = 0
            if len(date) >= 9:
                hour = int(date[7:9])
            if len(date) >= 11:
                minute = int(date[9:11])
            
            return datetime.datetime(year, month, day, hour, minute)


    # 输入20172521622 hour和minute可以不存在
    @classmethod
    def convert_julian_to_datetime(self, julian_date):
        return self.time_format_convert(julian_date, to_julian=False)


    #OR_ABI-L1b-RadC-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
    #OR_ABI-L1b-RadM1-M3C01_G16_s20172500027163_e20172500029536_c20172500029578.nc
    @classmethod
    def goes_file_name_seperation(self, file_name):
        file_name = os.path.split(file_name)[1]
        seg = file_name.split('_')

        sensor = seg[1][:12]
        channel = seg[1][-5:]

        year = seg[3][1:5]
        jd = seg[3][5:8]
        hour = seg[3][8:10]
        minute = seg[3][10:12]
        return (sensor, channel, year, jd, hour, minute, self.convert_julian_to_datetime(year+jd+hour+minute))



class GoesCache(HurricaneModel):
    def search_bands(self, sensor):
        if sensor == 'M3C01' or sensor =='VIS':
            return self.refVIS
        elif sensor == 'M3C14' or sensor == 'IR1':
            return self.tempIR1
        elif sensor == 'M3C15' or sensor == 'IR2':
            return self.tempIR2
        elif sensor == 'M3C09' or sensor == 'WV':
            return self.tempWV
        elif sensor == 'M3C07' or sensor == 'MWIR':
            return self.tempMWIR



class AwxFirstHeader(object):
    def __init__(self):
        self.str_file_name = 'DONTKNOW.AWX'
        self.byte_order = 1
        self.first_header_length = 40
        self.second_header_length = 64
        self.fill_section_length = -1
        self.recoder_length = -1
        self.recoder_nums_of_header = -1
        self.recoder_nums_of_data = -1
        self.type_product = 1
        self.type_compress = 0
        self.str_version = 'DONTKNOW'
        self.flag_quality = 0


class AWXSecondHeader(object):
    def __init__(self):
        self.sat_name = 'GOES-16\0'

        self.year = -1
        self.month = -1
        self.day = -1
        self.hour = -1
        self.minute = -1

        self.channel = -1
        self.projection_flag = 2
        self.img_height = -1
        self.img_width = -1
        self.left_top_scan = 0
        self.left_top_pixel = 0
        self.sample_ratio = 0

        self.north = -1
        self.south = -1
        self.west = -1
        self.east = -1
        self.center_projection_latitude = 0
        self.center_projection_longtitude = 0
        self.standard_latitude = 0
        self.standard_longitude = 0

        self.horizontal_resolution = -1
        self.vertical_resoulution = -1

        self.overlap_flag_geogrid = 0
        self.overlap_value_geogrid = 0
        self.color_table_length = 0
        self.calibration_table_length = 0
        self.geolocation_length = 0
        
        self.reserved = 0



def write_in_awx(first_header, second_header, img_data, save_name): 
    fp = open(save_name, 'wb')

    for c in first_header.str_file_name:
        fp.write( struct.pack('c', bytes(c, encoding='utf-8')) )
    fp.write( struct.pack('h', first_header.byte_order) )
    fp.write( struct.pack('h', first_header.first_header_length) )
    fp.write( struct.pack('h', first_header.second_header_length) )
    fp.write( struct.pack('h', first_header.fill_section_length) )
    fp.write( struct.pack('h', first_header.recoder_length) )
    fp.write( struct.pack('h', first_header.recoder_nums_of_header) )
    fp.write( struct.pack('h', first_header.recoder_nums_of_data) )
    fp.write( struct.pack('h', first_header.type_product) )
    fp.write( struct.pack('h', first_header.type_compress) )
    for c in first_header.str_version:
        fp.write( struct.pack('c', bytes(c, encoding='utf-8')) )
    fp.write( struct.pack('h', first_header.flag_quality) )

    for c in second_header.sat_name:
        fp.write( struct.pack('c', bytes(c, encoding='utf-8')) )
    fp.write( struct.pack('h', second_header.year) )
    fp.write( struct.pack('h', second_header.month) )
    fp.write( struct.pack('h', second_header.day) )
    fp.write( struct.pack('h', second_header.hour) )
    fp.write( struct.pack('h', second_header.minute) )
    fp.write( struct.pack('h', second_header.channel) )
    fp.write( struct.pack('h', second_header.projection_flag) )
    fp.write( struct.pack('h', second_header.img_height) )
    fp.write( struct.pack('h', second_header.img_width) )
    fp.write( struct.pack('h', second_header.left_top_scan) )
    fp.write( struct.pack('h', second_header.left_top_pixel) )
    fp.write( struct.pack('h', second_header.sample_ratio) )
    fp.write( struct.pack('h', int(second_header.north)) )
    fp.write( struct.pack('h', int(second_header.south)) )
    fp.write( struct.pack('h', int(second_header.west)) )
    fp.write( struct.pack('h', int(second_header.east)) )
    fp.write( struct.pack('h', second_header.center_projection_latitude) )
    fp.write( struct.pack('h', second_header.center_projection_longtitude) )
    fp.write( struct.pack('h', second_header.standard_latitude) )
    fp.write( struct.pack('h', second_header.standard_longitude) )
    fp.write( struct.pack('h', second_header.horizontal_resolution) )
    fp.write( struct.pack('h', second_header.vertical_resoulution) )
    fp.write( struct.pack('h', second_header.overlap_flag_geogrid) )
    fp.write( struct.pack('h', second_header.overlap_value_geogrid) )
    fp.write( struct.pack('h', second_header.color_table_length) )
    fp.write( struct.pack('h', second_header.calibration_table_length) )
    fp.write( struct.pack('h', second_header.geolocation_length) )
    fp.write( struct.pack('h', second_header.reserved) )

    for _ in range(first_header.fill_section_length):
        fp.write( struct.pack('c', bytes('0', encoding='utf-8')) )

    for d in img_data.flat:
        fp.write( struct.pack('f', d) )
    
    fp.close()




def save_to_awx(goes_cache, file_paths, dir_name):

    def fill_awx_header():
        # first header
        first_header.recoder_length = gc.shape[0]
        first_header.recoder_nums_of_data = gc.shape[1]

        tmp = first_header.first_header_length + first_header.second_header_length
        first_header.recoder_nums_of_header = math.ceil( tmp / first_header.recoder_length )
        first_header.fill_section_length = first_header.recoder_length - tmp

        # second header
        second_header.year = date.year
        second_header.month = date.month
        second_header.day = date.day
        second_header.hour = date.hour
        second_header.minute = date.minute

        if channel == 'M3C01':
            second_header.channel = 4
        elif channel == 'M3C14':
            second_header.channel = 1
        elif channel == 'M3C15':
            second_header.channel = 3
        elif channel == 'M3C09':
            second_header.channel = 2
        elif channel == 'M3C07':
            second_header.channel = 5

        second_header.img_height = gc.shape[0]
        second_header.img_width = gc.shape[1]

        second_header.horizontal_resolution = gc.shape[0]
        second_header.vertical_resoulution = gc.shape[1]

        second_header.north, second_header.south = np.max(gc.g16nc.latitude_table), np.min(gc.g16nc.latitude_table)
        second_header.east, second_header.west = np.max(gc.g16nc.longitude_table), np.min(gc.g16nc.longitude_table)

    gc = goes_cache
    for fp in file_paths:
        (_, basename) = os.path.split(fp)
        tmp = FileNameProcess.goes_file_name_seperation(basename)

        channel = tmp[1]
        date = tmp[6]

        first_header = AwxFirstHeader()
        second_header = AWXSecondHeader()

        fill_awx_header()
        img_data = gc.search_bands(channel)

        jn = os.path.splitext(basename)[0]
        write_in_awx(first_header, second_header, img_data, os.path.join(dir_name, jn+'.AWX'))



def load_directory(dirs):
    for d in dirs:
        file_names = os.listdir(d)
        full_names = []
        for f in file_names:
            full_names.append( os.path.join(d, f) )
        gc = GoesCache(files_name=full_names)
        save_to_awx(gc, full_names, 'D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\AWX_Data\\')
        print("save : %s" %(d))



if __name__ == "__main__":

    # def A():
    #     def B():
    #         print(j)
    #     for i in range(10):
    #         j = i + 1
    #         B()
    # A()    

    dirs = []
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M1-2531622\\')
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M1-2541600\\')
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M1-2721600\\')
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M2-2551559\\')
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M2-2581600\\')
    dirs.append('D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\M2-2721600\\')
    load_directory(dirs)

