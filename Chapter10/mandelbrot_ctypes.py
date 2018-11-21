from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *

# Linux users: change the filename below to './mandelbrot.so'
mandel_dll = CDLL('./mandelbrot.dll')
mandel_c = mandel_dll.launch_mandelbrot
mandel_c.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_int]


def mandelbrot(breadth, low, high, max_iters, upper_bound):

    lattice = np.linspace(low, high, breadth, dtype=np.float32)
    out = np.empty(shape=(lattice.size,lattice.size), dtype=np.float32)
    mandel_c(lattice.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)), c_int(max_iters), c_float(upper_bound), c_int(lattice.size) )  
    
    return out


if __name__ == '__main__':

    t1 = time()
    mandel = mandelbrot(512,-2,2,256, 2)
    t2 = time()

    mandel_time = t2 - t1
    
    print 'It took %s seconds to calculate the Mandelbrot graph.' % mandel_time

    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()


