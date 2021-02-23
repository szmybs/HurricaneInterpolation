import numpy as np
import math

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
