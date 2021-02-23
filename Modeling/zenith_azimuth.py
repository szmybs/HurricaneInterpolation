import numpy as np
import datetime
import math
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.goes_calculation import GOES
from GOES.goes import GOES_netCDF
from GOES.time_format import convert_datetime_to_julian


# g16nc 是一个 GOES_netCDF类的实例
def solar_satellite_zenith_azimuth_angle(g16nc):

    if hasattr(g16nc, 'longitude_table') == False or hasattr(g16nc, 'latitude_table') == False:
        latt, lont = GOES.compute_latitude_longitude_table(g16nc.data_head.y, g16nc.data_head.x, g16nc.data_head.geospatial_lon_nadir)
        g16nc.add_latitude_table(latt)
        g16nc.add_longitude_table(lont)


    data_date = convert_datetime_to_julian(g16nc.date)
    JD = int(data_date[1])
    utc_hour = int(data_date[2])
    minutes = int(data_date[3])

    delta = -23.45 * math.cos( (2*math.pi*(JD+10)) / 365 )

    beta = 2*math.pi * (JD-81) / 365
    Xtime = 9.87*math.sin(2*beta) - 7.53*math.cos(beta) - 1.5*math.sin(beta)
    l = utc_hour + minutes/60 + g16nc.longitude_table/15 + Xtime/60
    Ha = 15*(l-12)

    delta = np.deg2rad(delta)
    Ha = np.deg2rad(Ha)

    sinlat = np.sin(np.deg2rad(g16nc.latitude_table))
    coslat = np.cos(np.deg2rad(g16nc.latitude_table))
    sindelta = np.sin(delta)
    cosdelta = np.cos(delta)
    cosHa = np.cos(Ha)

    # solar zenith angle 太阳天顶角
    solar_zenith_angle = np.add( np.multiply(sinlat, sindelta), np.multiply( np.multiply(coslat, cosdelta), cosHa ) )
    solar_zenith_angle = np.arccos(solar_zenith_angle)

    # solar azimuth angle 太阳方位角
    solar_azimuth_angle = np.subtract( np.multiply(sindelta, coslat), np.multiply(np.multiply(cosHa, cosdelta), sinlat) )
    solar_azimuth_angle = np.divide(solar_azimuth_angle, np.sin(solar_zenith_angle))
    solar_azimuth_angle = np.arccos(solar_azimuth_angle)
    # solar azimuth angel 的不同算法
    # solar_azimuth_angle = np.subtract(sindelta, np.multiply(np.cos(solar_zenith_angle), sinlat))
    # solar_azimuth_angle = np.divide(solar_azimuth_angle, np.multiply(np.sin(solar_zenith_angle), coslat))
    # solar_azimuth_angle = np.arccos(solar_azimuth_angle)

    # 转换为角度
    # sza_d = np.degrees(solar_zenith_angle)
    # saa_d = np.degrees(solar_azimuth_angle)

    # 地球半径
    radius = 6378
    # 卫星轨道距地表距离
    satellite_dis = 35786
    dv = radius + satellite_dis

    longitude_dif = np.deg2rad(g16nc.longitude_table) - np.deg2rad(g16nc.geospatial_lon_nadir)
    center_angle = np.arccos( np.multiply( coslat, np.cos(longitude_dif) ) )

    satellite_azimuth_angle = np.divide( np.sin(longitude_dif), np.sin(center_angle) )
    satellite_azimuth_angle = np.arcsin(np.sin(satellite_azimuth_angle))

    be = np.divide( (np.cos(center_angle) - radius/dv), np.sin(center_angle) )
    be = np.arctan(be)
    satellite_zenith_angle = math.pi/2 - (center_angle + be)

    '''    
    sinlng = np.sin(np.deg2rad(g16nc.longitude_table))
    coslng = np.cos(np.deg2rad(g16nc.longitude_table))
    # 由于静止卫星在赤道上空，星下点纬度为0，因此只需计算星下点的经度
    sin_ndir = np.sin(np.deg2rad(g16nc.geospatial_lon_nadir))
    cos_ndir = np.cos(np.deg2rad(g16nc.geospatial_lon_nadir)) 

    re = ( radius*np.multiply(coslat, coslng), radius*np.multiply(coslat, sinlng), radius*sinlat )
    rs = ( dv*cos_ndir, dv*sin_ndir, 0 )
    rd = ( rs[0]-re[0], rs[1]-re[1], rs[2]-re[2] )

    def dot(x, y):
        tmp = 0
        for i in range(len(x)):
            tmp += np.multiply(x[i], y[i])
        return tmp
    rs_rd = dot(rs, rd)
    rs_dis = np.sqrt( dot(rs, rs) )
    rd_dis = np.sqrt( dot(rd, rd) )

    # satellite zenith angle 是正确的 azimuth 存在问题
    # satellite zenith angle
    satellite_zenith_angle = np.divide(rs_rd, np.multiply(rs_dis, rd_dis))
    satellite_zenith_angle = np.arccos(satellite_zenith_angle)
    
    rn = ( -np.multiply(sinlat, coslng), -np.multiply(sinlat, sinlng), coslat )
    re_dis = np.sqrt( dot(re, re) )

    def compute_rH():
        cos_sza = np.cos(satellite_zenith_angle)
        re_tmp = re
        for i in re_tmp:
            i = np.multiply( np.divide(i, re_dis), cos_sza )

        rd_re_tmp = dot(rd, re_tmp)
        rd_tmp = list(rd)
        for i in range(len(rd_tmp)):
            rd_tmp[i] = rd_tmp[i] - rd_re_tmp  #？
        return tuple(rd_tmp)
    rh = compute_rH()

    rh_rn = dot(rh, rn)
    rh_dis = np.sqrt( dot(rh, rh) )
    rn_dis = np.sqrt( dot(rn, rn) )

    # satellite azimuth angle
    satellite_azimuth_angle = np.divide(rh_rn, np.multiply(rh_dis, rn_dis))
    satellite_azimuth_angle = np.arccos(satellite_azimuth_angle)
    '''

    return solar_zenith_angle, solar_azimuth_angle, satellite_zenith_angle, satellite_azimuth_angle




if __name__ == "__main__":
    pass
    # path = 'D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\ABI-L1b-RadC\\M3C01\\2017253\\'
    # name = 'OR_ABI-L1b-RadC-M3C01_G16_s20172531622165_e20172531624538_c20172531624581.nc'

    # g16nc = GOES_netCDF(os.path.join(path, name))
    # solar_satellite_zenith_azimuth_angle(g16nc, 253, 16)

    # zero = np.ones(shape=(3,1))
    # one = np.ones(shape=(3,5))
    # print(zero + one)

    today = datetime.date(2020, 8, 26)
    today = today.strftime(("%j"))
    print(type(today))