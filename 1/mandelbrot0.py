from time import time
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np


def simple_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)
        
    # we will represent members as 1, non-members as 0.
    
    mandelbrot_graph = np.ones((height,width), dtype=np.float32)
    
    for x in range(width):
        
        for y in range(height):
            
            c = np.complex64( real_vals[x] + imag_vals[y] * 1j  )            
            z = np.complex64(0)
            
            for i in range(max_iters):
                
                z = z**2 + c
                
                if(np.abs(z) > upper_bound):
                    mandelbrot_graph[y,x] = 0
                    break
                
    return mandelbrot_graph


if __name__ == '__main__':
    
    t1 = time()
    mandel = simple_mandelbrot(512,512,-2,2,-2,2,256, 2.5)
    t2 = time()
    mandel_time = t2 - t1
    
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()
    
    dump_time = t2 - t1
    
    print 'It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time)
    print 'It took {} seconds to dump the image.'.format(dump_time)
    
    
    
    
    
    
