import numpy as np
import math


def weight2(a, b, c, d, e):
    # 获得中心车辆坐标
    x, y = b   # 获得目标车辆坐标
    x1, y1 = a  # 下一帧目标车坐标
    kv = 0.25
    kc = 2
    v0 = d
    v = e

    x0 = kv*v0*math.cos(c/180*math.pi)
    y0 = kv*v0*math.sin(c/180*math.pi)

    sigma1 = (1+kv*v0*math.cos(c/180*math.pi))*2
    sigma2 = (0.5+kv*v0*math.sin(c/180*math.pi))*2

    bf = x**2+(sigma1/sigma2)**2*y**2/(sigma1/sigma2)**2
    af = (sigma1/sigma2)**2*bf
    costheta = (x/af*(x1-x)+y/bf*(y1-y))/(((x/af)**2+(y/bf)**2)**0.5*((x1-x0)**2+(y1-y0)**2)**0.5)

    ca = 100
    s = ca*np.exp(-(x - x0) ** 2 / (sigma1 ** 2) - (y - y0) ** 2 / (sigma2 ** 2))  # 计算安全场场强大小
    R = s*np.exp(-kc*v*costheta)
    R = round(R, 5)
    return R