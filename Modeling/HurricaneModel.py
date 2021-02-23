import numpy as np
import os
import threading
import math
from enum import IntEnum
from scipy.optimize import leastsq

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GOES.goes import GOES_netCDF
from Modeling.zenith_azimuth import solar_satellite_zenith_azimuth_angle
from Modeling.WritingVTI import WritingVTI


debug_mode = True


class CloudType(IntEnum):
    Cloudless = 0
    WaterCloud = 1
    IceCloud = 2
    CirrusCloud = 3


class HurricaneModel(object):
    def __init__(self, files_name, **kwargs):
        if isinstance(files_name, list):
            self.g16nc = GOES_netCDF(files_name)
        else:
            rad = np.load(files_name)
            head, date = kwargs['Head'], kwargs['Date']
            self.g16nc = GOES_netCDF(Rad=rad, Head=head, Date=date)

        # shape
        self.shape = self.g16nc.shape

        # 波段
        self.IR1_wave_length = self.g16nc.wave_length[-2]
        self.WV_wave_length = self.g16nc.wave_length[-3]
        self.MWIR_wave_length = self.g16nc.wave_length[-4]
        self.VIS_wave_length = self.g16nc.wave_length[0]

        # Band 7 ~ 16 : 7,9,14,15
        self.tempEMI = self.conversion_radiances_to_brightness_temperature(self.g16nc.Rad[1:])
        self._convsersion_bands_nums()

        # 计算天顶角 方位角
        self.solar_zenith_angle, self.solar_azimuth_angle, self.satellite_zenith_angle, self.satellite_azimuth_angle = solar_satellite_zenith_azimuth_angle(self.g16nc)
       
        # Band 1 ~ 6 : only 1
        self.refVIS = self.conversion_radiances_to_reflectance(self.g16nc.Rad[0], self.g16nc.data_head.kappa0, self.solar_zenith_angle)

        # 地表温度
        self.lst = np.max(self.tempIR1)

        # 内在属性
        # 分别为粒子半径 非对称因子 可见光光学厚度 MWIR光学厚度 消光系数 单次散射反照率
        self.radius = 0
        self.asymmetry_para = 0
        self.vis_thickness = 0
        self.mwir_thickness = 0
        self.extinction_coef = 0
        self.albedo = 0


    def get_hurricane_model(self):
        self._cloud_classification()

        self._cloud_top_height()

        self._compute_water_cloud_property()
        self._compute_ice_cirrus_cloud_property()

        self._compute_water_cloud_extinction_coefficient()
        self._compute_ice_cirrus_cloud_extinction_coefficient()

        self._compute_cloud_thickness()

        self._export_volumetric_model(save_name=None)



    # 无云:0  水云:1  冰云:2  卷云:3
    # 这里使用一个按照优先级逆序，不断覆盖的方法
    def _cloud_classification(self):
        if debug_mode == True:
            cache_file = os.path.join(os.getcwd(), "Modeling/Cache/CloudType.npy")
            if os.path.exists(cache_file) == True:
                self.cloud_type = np.load(cache_file)
                self.has_cloud = np.where(self.cloud_type > 0, 1, 0)
                return
                
        condition1 = np.where(self.tempIR1 < 233, 1, 0)
        # condition2 = np.where(self.tempIR1 < (self.lst-2.5), 1, 0)
        condition2 = condition1     # !!!
        condition3 = np.where(self.refVIS > 0.45, 1, 0)

        # 这里1表示有云, 0表示无云
        self.has_cloud = np.where((condition1+condition2+condition3)>0, 1, 0)

        ice_cloud1 = np.where(condition1 > 0, CloudType.IceCloud, CloudType.Cloudless)
        water_cloud1 = np.where(self.tempIR1 > 273, CloudType.WaterCloud, CloudType.Cloudless)

        slope_map, _ = self._get_slope(x=self.tempIR1, y=self.tempIR2, scope=2)
        conditionK03 = np.where(slope_map > 0.3, 1, 0)

        ice_cloud2 = np.where(conditionK03 > 0, CloudType.IceCloud, CloudType.Cloudless)
        water_cloud2 = np.where(conditionK03 <= 0, CloudType.WaterCloud, CloudType.Cloudless)

        cirrus_conditionK = np.where( (slope_map > 0.3) & (slope_map < 0.5), 1, 0)
        cirrus_conditionT = np.where( (self.tempIR1 - self.tempIR2) > 1.4, 1, 0)
        cirrus_condition = np.multiply(conditionK03, cirrus_conditionK) + np.multiply(conditionK03, cirrus_conditionT)

        cirrus_cloud = np.where(cirrus_condition > 0, CloudType.CirrusCloud, CloudType.Cloudless)

        inverse_list = [cirrus_cloud, water_cloud2, water_cloud1, ice_cloud1]

        self.cloud_type = ice_cloud2
        for cloud in inverse_list:
            self.cloud_type = np.where(cloud > 0, cloud, self.cloud_type)
        self.cloud_type = np.multiply(self.cloud_type, self.has_cloud)

        if debug_mode == True:
            np.save(cache_file, self.cloud_type)


    def _cloud_top_height(self):
        if debug_mode == True:
            cache_file = os.path.join(os.getcwd(), "Modeling/Cache/Tct.npy")
            if os.path.exists(cache_file) == True:
                self.tct = np.load(cache_file)
                self.cloud_top_height = ((self.lst - self.tct) / 6.5) * 1000
                return

        # 冰云和水云区域的云顶温度
        tct1 = self.tempIR1

        # 计算卷云的云顶温度
        # 使用迭代法求解
        IR1_wave_length = self.IR1_wave_length
        WV_wave_length = self.WV_wave_length

        planck_IR1 = self._planck_function(self.tempIR1, IR1_wave_length)
        planck_WV = self._planck_function(self.tempWV, WV_wave_length)
        k, b = self._get_slope(x=planck_WV, y=planck_IR1, scope=2)

        # 理论上233K下被分类为冰云，273K下被分类为水云
        low = np.ones(shape = self.tempIR1.shape) * 233
        high = np.ones(shape = self.tempIR1.shape) * 273
        # 233 - 273 十次就足够了 准确说是十一次
        for _ in range(10):
            mid = (low + high) / 2
            m = self._planck_function(mid, IR1_wave_length) - ( np.multiply(k, self._planck_function(mid, WV_wave_length)) + b)

            low = np.where(m < 0, mid, low)
            high = np.where( m > 0, mid, high)
        tct2 = (low + high) / 2

        tct = np.multiply(tct1, np.where((self.cloud_type==1) | (self.cloud_type==2), 1, 0)) + np.multiply(tct2, np.where(self.cloud_type==3, 1, 0)) 
        self.tct = tct

        # 6.5是温度递减率6.5k/km  所以这里云顶高度的单位是米(m)
        self.cloud_top_height = ((self.lst - tct) / 6.5) * 1000

        if debug_mode == True:
            np.save(cache_file, np.asarray(self.tct))


    # 计算水云的属性
    # 本函数变量命名参照论文Simple approximate solutions of the radiative transfer equation for a cloudy atmosphere
    # 本函数公式同样来自上文
    # 为了近似求解这个超级复杂的方程组，采用的是均匀采样的方法，即在一定范围内试根，选择最接近的那个
    # 试根的范围选择来自物理参数的实际范围
    def _compute_water_cloud_property(self):
        refMWIR = self._conversion_MWIR_temperature_to_reflective()
        refVIS = self.refVIS
        MWIR_wave_length = self.MWIR_wave_length
        VIS_wave_length = self.VIS_wave_length

        # mu 是不是用取绝对值？
        mu = np.cos(self.satellite_zenith_angle)
        mu0 = np.cos(self.solar_zenith_angle)
        phi =  np.subtract(self.satellite_azimuth_angle, self.solar_azimuth_angle)
        mu_mul_mu0 = np.multiply(mu, mu0)
        mu_add_mu0 = np.add(mu, mu0)

        # 散射角 scattering angle
        theta = np.arccos(mu_mul_mu0 + np.multiply( np.multiply( np.sin(self.solar_zenith_angle), np.sin(self.satellite_zenith_angle) ), np.cos(phi) ))

        k0mu = 3/7 * (1 + 2*mu)
        k0mu0 = 3/7 * (1 + 2*mu0)

        minRe, maxRe, steps = 1, 30, 100
        re = np.arange(minRe, maxRe, (maxRe-minRe)/steps, np.float)
        re = np.expand_dims(re, axis=(1, 2))
        re = np.broadcast_to(re, shape=(re.shape[0], refVIS.shape[0], refVIS.shape[1]))       #这里shape待定

        def compute_thickness(radius, reflectance):
            # 非对称因子 asymmetry parameter 
            g = 0.809 + 3.387 * 0.001 * radius

            # Henyey-Greenstein phase function / 是否需要除以 4*pi ?
            phase_func = np.power(1 + np.multiply(g, g) - 2*np.multiply(g, np.cos(theta)), 1.5)
            phase_func = np.divide(1 - np.multiply(g, g), phase_func) #/ 4*np.pi

            # 几个经验常数系数
            A, B, C = 3.944, -2.5, 10.664
            rinf = np.divide( (A + B*mu_add_mu0 + C*mu_mul_mu0 + phase_func), 4*mu_add_mu0 )

            t = np.divide( (rinf - reflectance), np.multiply(k0mu, k0mu0) )
            thickness = np.divide( 1/t-1.07, 0.75*(1-g) )
            return thickness

        vis_thickness = compute_thickness(re, refVIS)

        def using_vis_compute_mwir_reflection():
            g = 0.726 + 6.652 * 0.001 * re

            # 在假定粒子半径的情况下，通过VIS的光学厚度计算MWIR的光学厚度
            tmp1 = np.power(2*np.pi*re/MWIR_wave_length, 2/3)
            tmp2 = np.power(2*np.pi*re/VIS_wave_length, 2/3)
            mwir_thickness = np.multiply(np.divide(1.1+tmp1, 1.1+tmp2), vis_thickness) * np.power(MWIR_wave_length*VIS_wave_length, 2/3)

            # 单次散射反照率 single scattering albedo 
            w = 1 - 0.025 - 0.0122 * re
            # 为什么吸收系数与消光系数之比等于 1-单次散射反照率， 我也不知道……
            beta = 1 - w
            x = np.multiply( np.square(np.multiply(3*(1-g), beta)), mwir_thickness)
            y = 4 * np.square( np.divide( beta, 3*(1-g) ) )

            t = np.divide( np.sinh(y), np.sinh(1.07*y + x) )
            delta = np.divide( np.multiply(4.86 - 13.08*mu_mul_mu0 + 12.76*np.square(mu_mul_mu0), np.exp(x)), np.power(mwir_thickness, 3) )
            tmd = t - delta

            # 同compute_vis_thickness()
            # Henyey-Greenstein phase function / 是否需要除以 4*pi ?
            phase_func = np.power(1 + np.multiply(g, g) - 2*np.multiply(g, np.cos(theta)), 1.5)
            phase_func = np.divide(1 - np.multiply(g, g), phase_func) #/ 4*np.pi

            # 几个经验常数系数
            A, B, C = 3.944, -2.5, 10.664
            rinf = np.divide( (A + B*mu_add_mu0 + C*mu_mul_mu0 + phase_func), 4*mu_add_mu0 )

            # 根据假定粒子半径计算出的mwir_reflectance
            ref_l = np.divide( np.multiply(np.multiply(-y, 1-0.05*y), mu_mul_mu0), rinf )
            ref_l = np.multiply(rinf, np.exp(ref_l))

            ref_r = np.multiply( np.multiply(tmd, np.exp(-x-y)), mu_mul_mu0)
            ref = ref_l - ref_r
            return ref, mwir_thickness, w

        fake_mwir_ref, mwir_thickness, albedo = using_vis_compute_mwir_reflection()
        error = np.abs(fake_mwir_ref - refMWIR)

        k = np.argmin(error, axis=0)
        i = np.multiply(np.ones(shape=self.shape), np.arange(0, self.shape[0], 1))        #这里不是arange不是self.shape[0]就是self.shape[1]
        i = i.astype(np.int)
        j = np.transpose(i)
        index = (k, j, i)

        # self.radius += np.where(self.cloud_type == CloudType.WaterCloud, fake_mwir_ref[index], 0)
        self.radius += np.where(self.cloud_type == CloudType.WaterCloud, re[index], 0)
        self.vis_thickness += np.where(self.cloud_type == CloudType.WaterCloud, vis_thickness[index], 0)
        self.mwir_thickness += np.where(self.cloud_type == CloudType.WaterCloud, mwir_thickness[index], 0)

        # 单次散射反照率按波段变化？
        self.albedo += np.where(self.cloud_type == CloudType.WaterCloud, albedo[index], 0)


    # 计算冰云和卷云的属性
    # 除了一些常数及增加一些属性的等价关系之外，与计算水云属性方法相同
    def _compute_ice_cirrus_cloud_property(self):
        refMWIR = self._conversion_MWIR_temperature_to_reflective()
        refVIS = self.refVIS
        MWIR_wave_length = self.MWIR_wave_length
        VIS_wave_length = self.VIS_wave_length

        # mu 是不是用取绝对值？
        mu = np.cos(self.satellite_zenith_angle)
        mu0 = np.cos(self.solar_zenith_angle)
        phi =  np.subtract(self.satellite_azimuth_angle, self.solar_azimuth_angle)
        mu_mul_mu0 = np.multiply(mu, mu0)
        mu_add_mu0 = np.add(mu, mu0)

        # 散射角 scattering angle
        theta = np.arccos(mu_mul_mu0 + np.multiply( np.multiply( np.sin(self.solar_zenith_angle), np.sin(self.satellite_zenith_angle) ), np.cos(phi) ))

        k0mu = 3/7 * (1 + 2*mu)
        k0mu0 = 3/7 * (1 + 2*mu0)

        minRe, maxRe, steps = 1, 30, 100
        re = np.arange(minRe, maxRe, (maxRe-minRe)/steps, np.float)
        re = np.expand_dims(re, axis=(1, 2))
        re = np.broadcast_to(re, shape=(re.shape[0], self.shape[0], self.shape[1]))       #这里shape待定

        def compute_thickness(reflectance):
            # 非对称因子 asymmetry parameter 
            g = 0.74

            # Henyey-Greenstein phase function / 是否需要除以 4*pi ?
            phase_func = np.power(1 + np.multiply(g, g) - 2*np.multiply(g, np.cos(theta)), 1.5)
            phase_func = np.divide(1 - np.multiply(g, g), phase_func) #/ 4*np.pi

            # 几个经验常数系数
            A, B, C = 1.247, 1.186, 5.157
            rinf = np.divide( (A + B*mu_add_mu0 + C*mu_mul_mu0 + phase_func), 4*mu_add_mu0 )

            t = np.divide( (rinf - reflectance), np.multiply(k0mu, k0mu0) )
            thickness = np.divide( 1/t-1.07, 0.75*(1-g) )
            return thickness

        vis_thickness = compute_thickness(refVIS)

        def using_vis_compute_mwir_reflection():
            g = 0.74

            # 在假定粒子半径的情况下，通过VIS的光学厚度计算MWIR的光学厚度
            tmp1 = np.power(2*np.pi*re/MWIR_wave_length, 2/3)
            tmp2 = np.power(2*np.pi*re/VIS_wave_length, 2/3)
            mwir_thickness = np.multiply(np.divide(1.1+tmp1, 1.1+tmp2), vis_thickness) * np.power(MWIR_wave_length*VIS_wave_length, 2/3)

            # 单次散射反照率 single scattering albedo 
            w = 1 - 0.025 - 0.0122 * re
            # 为什么吸收系数与消光系数之比等于 1-单次散射反照率， 我也不知道……
            beta = 1 - w
            x = np.multiply( np.square(np.multiply(3*(1-g), beta)), mwir_thickness)
            y = 4 * np.square( np.divide( beta, 3*(1-g) ) )

            t = np.divide( np.sinh(y), np.sinh(1.07*y + x) )
            delta = np.divide( np.multiply(4.86 - 13.08*mu_mul_mu0 + 12.76*np.square(mu_mul_mu0), np.exp(x)), np.power(mwir_thickness, 3) )
            tmd = t - delta

            # 同compute_vis_thickness()
            # Henyey-Greenstein phase function / 是否需要除以 4*pi ?
            phase_func = np.power(1 + np.multiply(g, g) - 2*np.multiply(g, np.cos(theta)), 1.5)
            phase_func = np.divide(1 - np.multiply(g, g), phase_func) #/ 4*np.pi

            # 几个经验常数系数
            A, B, C = 1.247, 1.186, 5.157
            rinf = np.divide( (A + B*mu_add_mu0 + C*mu_mul_mu0 + phase_func), 4*mu_add_mu0 )

            # 根据假定粒子半径计算出的mwir_reflectance
            # 这里不再使用 1-0.05y 很奇怪
            ref_l = np.divide( np.multiply(-y, mu_mul_mu0), rinf )
            ref_l = np.multiply(rinf, np.exp(ref_l))

            ref_r = np.multiply( np.multiply(tmd, np.exp(-x-y)), mu_mul_mu0)
            ref = ref_l - ref_r
            return ref, mwir_thickness, w

        fake_mwir_ref, mwir_thickness, albedo = using_vis_compute_mwir_reflection()
        error = np.abs(fake_mwir_ref - refMWIR)

        k = np.argmin(error, axis=0)
        i = np.multiply(np.ones(shape=self.shape), np.arange(0, self.shape[0], 1))        #这里不是arange不是self.shape[0]就是self.shape[1]
        i = i.astype(np.int)
        j = np.transpose(i)
        index = (k, j, i)

        # self.radius += np.where( (self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), fake_mwir_ref[index], 0)
        self.radius += np.where( (self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), re[index], 0)
        self.vis_thickness += np.where( (self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), vis_thickness, 0)
        self.mwir_thickness += np.where( (self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), mwir_thickness[index], 0)

        # 单次散射反照率按波段变化？
        self.albedo += np.where( (self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), albedo[index], 0)

    
    def _compute_cloud_thickness(self):
        self.radius = np.where(self.radius < 0, 0, self.radius)
        self.vis_thickness = np.where(self.vis_thickness < 0, 0, self.vis_thickness)
        self.mwir_thickness = np.where(self.mwir_thickness < 0, 0, self.mwir_thickness)
        self.extinction_coef = np.where(self.extinction_coef < 0, 0, self.extinction_coef)
        self.albedo = np.where(self.albedo < 0, 0, self.albedo)

        # 云的几何厚度
        self.geo_thickness = np.divide(self.vis_thickness, self.extinction_coef + 1e-9)
        self.geo_thickness = np.where(self.geo_thickness < 0, 0, self.geo_thickness)


    # 根据原代码使用LWP和LWC计算消光系数 - flag == 1
    # 根据论文公式计算消光系数 - flag == 0
    def _compute_water_cloud_extinction_coefficient(self, flag = 1):
        if flag == 0:
            n0 = 6e7
            radius = self.radius * 1e-6
            ec = 0.75 * n0 * np.pi * np.power(radius, 2)
            self.extinction_coef += np.where(self.cloud_type == CloudType.WaterCloud, ec, 0)
        elif flag == 1:
            rho = 1e3
            MWIR_wave_length = self.MWIR_wave_length
            radius = self.radius * 1e-6
            mwir_thickness = self.mwir_thickness

            # LWP 是 液水路径(Liquid Water Path) 虽然我也不知道这是什么
            LWP_1 = np.multiply( 2/3 * mwir_thickness * rho, radius )
            LWP_2 = 1 + 1.1 / np.power( 2*np.pi/MWIR_wave_length * radius , 2/3)
            LWP = np.divide(LWP_1, LWP_2)

            # 这个看起来是光速的样子 为什么这样命名
            NO = 3e8
            # 这个看起来像是水云谱分布的方差
            alpha = 2

            rn = radius / (alpha + 2)
            V = 4 / 3*np.pi * NO * np.power(rn, 3) * 24
            # LWC 是 液态水含量(Liquid Water Content)
            LWC = rho * V

            ec = np.divide(1.5 * LWC, rho * radius + 1e-9) 
            self.extinction_coef += np.where(self.cloud_type == CloudType.WaterCloud, ec, 0)
            # print("ec1")


    def _compute_ice_cirrus_cloud_extinction_coefficient(self):
        a0 = -6.656 * 1e-3
        a1 = 3.686

        tc = np.where(self.tct > 213, self.tct, 213) + 3.33 * self.vis_thickness
        tc = np.where(tc < 253, tc, 253)

        # IWC 是 冰水含量(ice water content)
        # 这里是2.445 还是 -2.445
        IWC = np.exp(-0.2443 * 1e3 * np.power(253 - tc, 2.445))
        IWC = np.exp(4 * IWC - 7.6)

        radius = self.radius * 1e-6
        # 这里是2还是1.1
        ec = IWC * ( a0 + a1/(1.1*radius + 1e-9) )
        # self.extinction_coef += np.where((self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), ec, 0)
        self.extinction_coef += np.where((self.cloud_type == CloudType.IceCloud) | (self.cloud_type == CloudType.CirrusCloud), 0.025, 0)
        # print("ec2")


    def _get_slope(self, x, y, scope):
        assert x.ndim == 2 and y.ndim == 2 and x.shape == y.shape

        slope_map = np.zeros(shape=x.shape)
        offset_map = np.zeros(shape=x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                
                if self.has_cloud[i][j] <= 0:
                    continue
                up, left = max(i-scope, 0), max(j-scope, 0)
                down, right = i+scope, j+scope
                # IR1_child = self.tempIR1[up:down+1][left:right+1]
                # WV_child = self.tempWV[up:down+1][left:right+1]

                x_child = x[up:down+1, left:right+1]
                y_child = y[up:down+1, left:right+1]

                k, b = self._least_square(x_child, y_child)
                slope_map[i][j] = k
                offset_map[i][j] = b

        return slope_map, offset_map

    def _least_square(self, x, y): 
        x = x.flatten()
        y = y.flatten()
    
        fit_ret = leastsq(lambda w,tx,ty: (w[0]*tx + w[1]) - ty, [0,0], args=(x, y))
        # print(type(fit_ret))
        k, b = fit_ret[0]
        # print("k = %f ;  b = %f" % (k,b))
        return k, b

   
    def _planck_function(self, t, wave_length):
        # wave_length 的单位是 米(m)
        c, h, k = 3e8, 6.626e-34, 1.3806e-23
        L = (2*h*c*c) / (np.power(wave_length, 5) * (np.exp( (h*c) / (wave_length*k*t) ) - 1))
        return L
    
    def _inverse_planck_function(self, L, wave_length):
        c, h, k = 3e8, 6.626e-34, 1.3806e-23

        A = 2 * h * c**2 / np.power(wave_length, 5)
        B = (h * c) / (wave_length * k)

        t = np.divide( B, np.log( np.divide(A, L) + 1 ) )
        return t


    def _conversion_MWIR_temperature_to_reflective(self):
        strange_constant = 0.9862e6 / math.pi
        MWIR_wave_length = self.g16nc.wave_length[-4]

        MWIR_pl = self._planck_function(self.tempMWIR, MWIR_wave_length)
        IR1_pl = self._planck_function(self.tempIR1, MWIR_wave_length)

        refMWIR =  np.divide( (MWIR_pl-IR1_pl), (strange_constant * np.cos(self.solar_zenith_angle) - IR1_pl) )
        return refMWIR


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


    def _convsersion_bands_nums(self):
        # VIS-M3C01  IR1-M3C14  IR2-M3C15  WV-M3C09  MWIR-M3C07
        # M3C01-VIS  M3C07-MWIR  M3C09-WV  M3C14-IR1  M3C15-IR2
        self.tempIR1 = self.tempEMI[-2]
        self.tempIR2 = self.tempEMI[-1]
        self.tempWV = self.tempEMI[-3]
        self.tempMWIR = self.tempEMI[-4]


    def _export_particles_model(self, save_name):
        pass

    def _export_volumetric_model(self, save_name):
        zaxis = 10
        vol = np.ones(shape=(self.shape[0], self.shape[1], zaxis), dtype=np.float32)

        # 几何厚度归一化
        cloud_bottom_height = self.cloud_top_height - self.geo_thickness
        min_bottom, max_top = np.amin(cloud_bottom_height), np.amax(self.cloud_top_height)

        height_scale = max_top - min_bottom

        # norm_thickness = self.geo_thickness / height_scale
        norm_bottom = (cloud_bottom_height - min_bottom) / height_scale
        norm_top = (self.cloud_top_height  - min_bottom) / height_scale

        # 消光系数归一化
        max_ext, min_ext = np.amax(self.extinction_coef), np.amin(self.extinction_coef)
        norm_ext = self.extinction_coef / (max_ext - min_ext)

        # 得到消光系数体
        mask = np.arange(start=0, stop=zaxis, step=1)
        mask = np.broadcast_to(mask, shape=vol.shape)

        tmp1 = np.expand_dims( np.floor(norm_bottom*zaxis), axis=2).repeat(vol.shape[2], 2)
        tmp2 = np.expand_dims( np.ceil(norm_top*zaxis), axis=2).repeat(vol.shape[2], 2)
        mask = np.where( (mask >= tmp1) & (mask < tmp2), 1, 0)

        vol = np.multiply(vol, np.expand_dims(norm_ext, axis=2).repeat(vol.shape[2], 2))
        # vol = np.multiply(vol, mask)

        WritingVTI(data=vol, save_name=os.path.join(os.getcwd(), "Modeling/XMLTest-nomask.vti"))
        # WritingVTI(data=vol, save_name=os.path.join(os.getcwd(), "Modeling/XMLTest.vti"))


        

if __name__ == "__main__":
    path = "D:\\Code\\GOES-R-2017-HurricaneExtraction\\Data\\OR_ABI-L1b-RadM1-253\\"
    file_names = os.listdir(path)

    full_names = []
    for i in file_names:
        full_names.append( os.path.join(path, i) )
    
    hm = HurricaneModel(files_name=full_names)
    hm.get_hurricane_model()

    print("OVER")

    # 最小二乘测试
    # train_x = np.array([160,165,158,172,159,176,160,162,171])
    # train_y = np.array([58,63,57,65,62,66,58,59,62])

    # k, b = HurricaneModel._least_square(train_x, train_y)

    # k = np.array([[7, 1], [37, 15]])
    # b = -np.array([[13, 9], [1480, 777]])

    # low = np.ones(shape = k.shape) * -100
    # high = np.ones(shape = k.shape) * 100
    # for _ in range(100):
    #     mid = (low + high) / 2
    #     m = np.multiply(k, mid) + b

    #     low = np.where(m < 0, mid, low)
    #     high = np.where( m > 0, mid, high)
    # tct2 = (low + high) / 2
    # print(tct2)
    # pass

    # broadcast_to 实验
    # a = np.arange(1, 10, 1)
    # print(a)
    # print(a.shape)
    # a = np.expand_dims(a, 0)
    # a = np.broadcast_to(a, (9,9))
    # print(a)
    # print(a.shape)

    # a = np.arange(1, 10, 1)
    # a = np.ones(shape=(9,))
    # a = np.reshape(a, newshape=(3,3))

    # re = np.arange(2,4,1)
    # # re = np.expand_dims(re, axis=(1,2))
    # re = np.broadcast_to(re, (2, 3, 3))
    # print(re)

    # # argmin 实验
    # a = np.arange(1, 28, 1)
    # a = np.reshape(a, newshape=(3,3,3))
    # # print(a)

    # k = np.argmin(a, axis=0)
    # # print(index)

    # i = np.ones(shape=(3,3))
    # mul = np.arange(0, 3, 1)
    # i = np.multiply(i, mul)
    # i = i.astype(np.int)
    # j = np.transpose(i)
    # # print(i)
    # # print(j)

    # index = np.stack((j, i, k), axis=2)
    # # print(index)
    # new_index = (k, j, i)
    # value = a[new_index]
    # print(value)

