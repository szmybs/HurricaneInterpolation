from netCDF4 import Dataset
import numpy as np
import os
import math
import datetime

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())   

from DataSet.sunrise_sunset import SunriseSunset
from DataSet.normalize import Normalize, Quantization


GOES_CHANNELS = ('M3C01', 'M3C07', 'M3C09', 'M3C14', 'M3C15')


def time_format_convert(date, to_julian=True):
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


def cut_filename(f):
    seg = f.split('_')
    sensor = seg[1][:-6]
    channel = seg[1][-5:]
    date = seg[4][1:8]
    return (sensor, channel, date)


class BestTrack(object):
    def __init__(self, file):
        try:
            f = open(file, 'r')
        except OSError:
            print("文件出错啦！")
        else:
            self.file_content = f.readlines()
            self.data = []
            f.close()
            self.post_processing()

    def process_header_line(self, num):
        header = self.file_content[num]
        header = header.replace(" ","")
        header = header.split(",")
        header_dict = {'Sign':header[0], 'Name':header[1], 'Lines':int(header[2])}
        return header_dict

    def process_data_line(self, num):
        data = self.file_content[num]
        data = data.replace(" ","")
        data = data.split(",")

        date = time_format_convert(data[0])[0] + data[1]
        n_lati = float(data[4][:-1])
        w_longi = float(data[5][:-1])

        if data[4][-1] == 'S' or data[4][-1] == 's':
            n_lati = -1 * n_lati
        if data[5][-1] == 'W' or data[5][-1] == 'w':
            w_longi = -1 * w_longi

        data_dict = {'Date':date, 'Latitude':n_lati, 'Longitude':w_longi}
        return data_dict

    def post_processing(self):
        extrinsic_cycle = 0
        while extrinsic_cycle < len(self.file_content):
            header = self.process_header_line(extrinsic_cycle)
            
            one_hurricane = []
            intrinsic_cycle = 1
            while intrinsic_cycle <= header['Lines']:
                data = self.process_data_line(extrinsic_cycle+intrinsic_cycle)
                data['Name'] = header['Name']
                one_hurricane.append(data)
                intrinsic_cycle += 1
            self.data.append(one_hurricane)
            extrinsic_cycle += intrinsic_cycle
    
    def get_track_data(self):
        return self.data
    
    def find_hurricane_location(self, time, interplotation_method=None):
        if interplotation_method is not None:
            print("暂时不支持插值方法选择")
            interplotation_method = None
        
        def interplotate(last, cur, time):
            data1 = (float(last['Date']), last['Latitude'], last['Longitude'])
            data2 = (float(cur['Date']), cur['Latitude'], cur['Longitude'])
            data3 = float(time)

            if interplotation_method is not None:
                loc = interplotation_method(data1, data2 ,data3)
            else:
                lat = ((data2[1] - data1[1]) / (data2[0] - data1[0])) * (data3 - data1[0]) + data1[1]
                lon = ((data2[2] - data1[2]) / (data2[0] - data1[0])) * (data3 - data1[0]) + data1[2]
                loc = (lat, lon)
            return loc

        hur_eye = []
        for hur in self.data:
            if time < hur[0]['Date'] or time > hur[-1]['Date']:
                hur_eye.append({})
                continue
            last = hur[0] 
            for ht in hur:
                if time == ht['Date']:
                    loc = (ht['Latitude'], ht['Longitude'])
                    hur_eye.append({'Location': loc, 'Name': ht['Name']})
                    break
                elif time > ht['Date']:
                    last = ht
                elif time < ht['Date']:
                    loc = interplotate(last, ht, time)
                    hur_eye.append({'Location': loc, 'Name': ht['Name']})
                    break
        return hur_eye


class PathSet(object):
    def __init__(self, root_path):
        self.root_path = root_path
        if os.path.isdir(self.root_path) is False:
            return
        self.path_tree = self.build_path_tree(self.root_path)
        #print("A")
        #self.offer_url()

    def build_path_tree(self, path):
        cets = sorted(os.listdir(path))
        pdict = {}
        for cet in cets:
            cet_path = os.path.join(path, cet)
            if os.path.isdir(cet_path) is True:
                tmp = self.build_path_tree(cet_path)
                pdict[cet] = tmp
        if len(pdict)<=0:
            return path
        return pdict

    def offer_url(self, select_date=None):
        m3cs = list(self.path_tree.keys())
        if len(m3cs) <=0:
            return
        dates = list(self.path_tree[m3cs[0]].keys())
        if select_date is not None and select_date in dates:
            dates.clear()
            dates.append(select_date)

        first_dir = []
        for date in dates:
            second_dir = []
            for m3c in m3cs:
                if (date in self.path_tree[m3c]) is False:
                    second_dir.clear()
                    break
                second_dir.append(self.path_tree[m3c][date])
            if len(second_dir) > 0:
                first_dir.append(second_dir)

        def get_files(dir):        
            files_tree = []
            for d in dir:
                files = sorted(os.listdir(d))
                files_dict = {}
                for f in files:
                    time = self.cut_filename(f)
                    files_dict[time] = os.path.join(d, f)
                files_tree.append(files_dict)
            
            ref = list(files_tree[0])
            for time in ref:
                files_path = []
                for files_dict in files_tree:
                    if time in files_dict:
                        files_path.append([time, files_dict[time]])
                    else:
                        files_path.clear()
                        break
                yield(files_path)
        
        for second_dir in first_dir:
            files_path = get_files(second_dir)
            while True:
                try:
                    nc = next(files_path)
                    yield(nc)
                except StopIteration:
                    break
            
    def cut_filename(self, f):
        seg = f.split('_')
        time = seg[3][1:12]
        return time


class VisibleLight(object):
    def __init__(self, date, latitude, longitude):
        self.date = date
        self.latitude = latitude
        self.longitude = longitude
   
    def convert_datetime_to_julian(self, date):
        return date.strftime("%Y%j%H%M")

    def convert_julian_to_datetime(self, julian_date):
        return time_format_convert(julian_date, to_julian=False)

    def isVisibility(self):
        if isinstance(self.date, str):
            date = self.convert_julian_to_datetime(self.date)
        elif isinstance(self.date, datetime.datetime):
            date = self.date
        
        ro = SunriseSunset(dt=date, latitude=self.latitude, longitude=self.longitude)
        sunrise_time, sunset_time = ro.calculate()
        #print("sunrise : %s - sunset : %s" % (sunrise_time, sunset_time))

        # day is out of range for month  August 31th
        if sunset_time.__le__(sunrise_time):
            sunset_time = sunset_time + datetime.timedelta(days=1)

        if sunrise_time.__le__(date) and date.__le__(sunset_time):
            return True
        return False


class HurricaneExtraction(object):
    def __init__(self, hur_data_path, hur_track_file, save_path='./', select_date=None):
        self.path_set = PathSet(hur_data_path)
        self.hur_url = self.path_set.offer_url(select_date)

        self.best_track = BestTrack(hur_track_file)
        self.hur_track = self.best_track.get_track_data()

        self.save_path = save_path
        #print(type(self.hur_track[0][0]['Date']))
    

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
    

    # section = (纬度, 经度)  &  网格点以2km为标准 
    def hurricane_extraction(self, section=(224, 224)):
        def get_elevation_scanning_angle(lamb0):
            es_angle = []
            for hl in hur_loc:
                if len(hl) <= 0:
                    continue
                tmp = self.navigating_from_geodetic_to_elevation_scanning_angle(hl['Location'][0], hl['Location'][1], lamb0)
                es_angle.append({'Location': tmp, 'Name': hl['Name']})
            return es_angle
        
        # 不再通过经纬度确定边界，实测会出现 一个很小的负数 不小于0的情况 使用<0
        def judge_bound(center, extract_bound, image_bound):
            y_image_bound = image_bound[0]
            x_image_bound = image_bound[1]

            (north, south) = (center[0] - extract_bound[0], center[0] + extract_bound[0])
            (east, west) = (center[1] + extract_bound[1], center[1] - extract_bound[1])
            
            yn = y_image_bound[0] - north
            ys = y_image_bound[1] - south
            if extract_bound[0] < 0:
                if yn < 0 or ys > 0:
                    return False
            else:
                if yn > 0 or ys < 0:
                    return False
            
            xe = x_image_bound[0] - east
            xw = x_image_bound[1] - west
            if extract_bound[1] < 0:
                if xe < 0 or xw > 0:
                    return False
            else:
                if xe > 0 or xw < 0:
                    return False
            
            return True
        
        def judge_visibility():
            visibility = {}
            for hl in hur_loc:
                if len(hl) <= 0:
                    continue
                vl = VisibleLight(date=time, latitude=hl['Location'][0], longitude=hl['Location'][1])
                visibility[hl['Name']] = vl.isVisibility()
            return visibility

        section = np.asarray(section)
        while True:
            try:
                data_list = next(self.hur_url)
                if len(data_list) <= 0:
                    continue

                time = data_list[0][0]
                hur_loc = self.best_track.find_hurricane_location(time)
                visibility = judge_visibility()

                hur_extraction_data = {}
                ex_angle = []
                for dl in data_list:
                    g16nc = Dataset(dl[1])
                    size = np.asarray ( (g16nc.variables['y'].size, g16nc.variables['x'].size) )
                    section_scaled = np.multiply( section, np.divide(size, np.asarray((1500, 2500))) )
                    section_scaled = section_scaled.astype(np.int32)

                    if len(ex_angle) <= 0:
                        ex_angle = get_elevation_scanning_angle(g16nc.variables['geospatial_lat_lon_extent'].geospatial_lon_nadir)
                        y_image_bound = np.asarray(g16nc.variables['y_image_bounds'][:])
                        x_image_bound = np.asarray(g16nc.variables['x_image_bounds'][:])
                    
                    for ex in ex_angle:
                        ex_loc = ex['Location']
                        ex_name = ex['Name']

                        hur_center = np.asarray(ex_loc)
                        #scale_factor = np.asarray( (g16nc.variables['y'].scale_factor, g16nc.variables['x'].scale_factor) )
                        
                        y_eye_grid = ((hur_center[0] - y_image_bound[0]) / (y_image_bound[1] - y_image_bound[0])) * size[0]
                        x_eye_grid = ((hur_center[1] - x_image_bound[0]) / (x_image_bound[1] - x_image_bound[0])) * size[1]
                        eye_grid = np.asarray( (y_eye_grid, x_eye_grid), dtype=np.int32)

                        north_west = (eye_grid - (section_scaled / 2)).astype(np.int32)
                        south_east = (eye_grid + (section_scaled / 2)).astype(np.int32)

                        if north_west[0] < 0 or north_west[1] < 0 or south_east[0] > size[0] or south_east[1] > size[1]:
                            print("台风不在观测区域内: %s - %s - %s" % (ex_name, time, dl[1]))
                            continue
                        
                        rad = g16nc.variables['Rad'][ north_west[0]: south_east[0], north_west[1]: south_east[1] ]
                        rad = np.ma.fix_invalid(rad, fill_value=rad.min)

                        if ex_name in hur_extraction_data:
                            hur_extraction_data[ex_name].append(rad)
                        else:
                            hur_extraction_data[ex_name] = []
                            hur_extraction_data[ex_name].append(rad)
                        #print('A')

                    g16nc.close()
                self.save_extraction_data(hur_extraction_data, time, visibility)
            
            except StopIteration:
                break
            except OSError:
                print('有文件出错啦:%s' % str(dl[1]))
                continue


    def save_extraction_data(self, hur_extraction_data, time, visibility):
        hur_names = list(hur_extraction_data.keys())

        for hur_name in hur_names:
            data = hur_extraction_data[hur_name]
            if len(data) < len(GOES_CHANNELS):
                print("缺少数据: %s" % (str(hur_name)))
                continue

            # 与 GOES_CHANNELS 吻合
            if visibility[hur_name] is False:
                path = os.path.join(self.save_path, hur_name, 'Invisible', time[:7])
                if os.path.exists(path) == False or os.path.isdir(path) == False:
                    os.makedirs(path)
                file_path = os.path.join(path, time+'_Invis')
            else:
                path = os.path.join(self.save_path, hur_name, 'Visible', time[:7])
                if os.path.exists(path) == False or os.path.isdir(path) == False:
                    os.makedirs(path)
                file_path = os.path.join(path, time)

            data = Quantization.convert_float_to_unsigned(data)

            np.savez(file=file_path, M3C01=data[0], M3C07=data[1], M3C09=data[2], M3C14=data[3], M3C15=data[4])
            print('save to %s' %(file_path))


    @classmethod
    def read_extraction_data(self, file):
        data_set = np.load(file)

        data = []
        data.append(data_set['M3C01'])
        data.append(data_set['M3C07'])
        data.append(data_set['M3C09'])
        data.append(data_set['M3C14'])
        data.append(data_set['M3C15'])
        
        return data


class HurricaneExtractionRadM(HurricaneExtraction):
    def hurricane_extraction(self, section=None):

        def get_elevation_scanning_angle(lamb0):
            es_angle = []
            for hl in hur_loc:
                if len(hl) <= 0:
                    continue
                tmp = self.navigating_from_geodetic_to_elevation_scanning_angle(hl['Location'][0], hl['Location'][1], lamb0)
                es_angle.append({'Location': tmp, 'Name': hl['Name']})
            return es_angle

        def judge_visibility():
            visibility = {}
            for hl in hur_loc:
                if len(hl) <= 0:
                    continue
                vl = VisibleLight(date=time, latitude=hl['Location'][0], longitude=hl['Location'][1])
                visibility[hl['Name']] = vl.isVisibility()
            return visibility

        while True:
            try:
                data_list = next(self.hur_url)
                if len(data_list) <= 0:
                    continue

                time = data_list[0][0]
                hur_loc = self.best_track.find_hurricane_location(time)
                visibility = judge_visibility()

                hur_extraction_data = {}
                ex_angle = []
                hur_name = ""
                for dl in data_list:
                    g16nc = Dataset(dl[1])

                    if len(ex_angle) <= 0:
                        ex_angle = get_elevation_scanning_angle(g16nc.variables['geospatial_lat_lon_extent'].geospatial_lon_nadir)
                        y_image_bound = np.asarray(g16nc.variables['y_image_bounds'][:])
                        x_image_bound = np.asarray(g16nc.variables['x_image_bounds'][:])

                        if (y_image_bound == np.array((-999,-999))).all() or (x_image_bound == np.array((-999,-999))).all():
                            g16nc.close()
                            print("文件出错了: %s " % (time))
                            break

                        size = np.asarray ( (g16nc.variables['y'].size, g16nc.variables['x'].size) )
                        min_distance = np.sum(size)

                        for ex in ex_angle:
                            ex_loc = ex['Location']
                            ex_name = ex['Name']

                            hur_center = np.asarray(ex_loc)

                            y_eye_grid = ((hur_center[0] - y_image_bound[0]) / (y_image_bound[1] - y_image_bound[0])) * size[0]
                            x_eye_grid = ((hur_center[1] - x_image_bound[0]) / (x_image_bound[1] - x_image_bound[0])) * size[1]
                            eye_grid = np.asarray( (y_eye_grid, x_eye_grid), dtype=np.int32)
                            
                            dist = np.sum(np.abs(np.subtract(eye_grid, size/2)))
                            if dist < min_distance:
                                min_distance = dist
                                hur_name = ex_name
                        if min_distance > np.sum(size/4):
                            g16nc.close()
                            print("这不是一个台风: %s - %d" % (time, min_distance))
                            break

                    rad = g16nc.variables['Rad'][:]
                    rad = np.ma.fix_invalid(rad, fill_value=rad.min)

                    if hur_name in hur_extraction_data:
                        hur_extraction_data[hur_name].append(rad)
                    else:
                        hur_extraction_data[hur_name] = []
                        hur_extraction_data[hur_name].append(rad)
                        #print('A')
                    g16nc.close()
                self.save_extraction_data(hur_extraction_data, time, visibility)
            
            except StopIteration:
                break
            except OSError:
                print('有文件出错啦:%s' % str(dl[1]))
                continue


if __name__ == "__main__":
    # tfc = time_format_convert('20202180622', to_julian=False)
    # vl = VisibleLight(date='20202180019', latitude=39.1068, longitude=-94.566)
    # vli = vl.isVisibility()

    # hur_data_path = './Data/ABI-L1b-RadC/'
    # best_track_file = './Data/best-track/2017-4hurricane-best-track.txt'

    # he = HurricaneExtraction(hur_data_path, best_track_file, './Data/NpyData/', select_date=None)
    # he.hurricane_extraction(section=(256, 256))

    # jd = time_format_convert('20170910')  #253
    # print(jd[0])

    hur_data_path = '/GOES-16-Data/DATA/ABI-L1b-RadM/'
    best_track_file = './DataSet/2017-4hurricane-best-track.txt'

    he = HurricaneExtractionRadM(hur_data_path, best_track_file, './DataSet/Data-RadM/', select_date=None)
    he.hurricane_extraction()