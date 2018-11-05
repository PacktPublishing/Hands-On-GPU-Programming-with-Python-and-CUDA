import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

# this is a work-efficent parallel prefix-sum algorithm.
# written by Brian Tuomanen for "Hands On GPU Programming with Python and CUDA"

# kernel for up-sweep phase
up_ker = SourceModule("""
__global__ void up_ker(double *x, double *x_old, int k )
{
     int j =  blockIdx.x*blockDim.x + threadIdx.x;
     
     int _2k = 1 << k;
     int _2k1 = 1 << (k+1);
     
     if (j % _2k1 == 0)
         x[j + _2k1 - 1] = x_old[j + _2k -1 ] + x_old[j + _2k1 - 1];

}
""")

up_gpu = up_ker.get_function("up_ker")

# implementation of up-sweep phase
def up_sweep(x):
    # let's typecast to be safe.
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x) )
    x_old_gpu = x_gpu.copy()
    for k in range( int(np.log2(x.size) ) ) : 
        up_gpu(x_gpu, x_old_gpu, np.int32(k)  , block=(32,1,1), grid=(int(x.size)/32,1,1))
        x_old_gpu[:] = x_gpu[:]
        
    x_out = x_gpu.get()
    return(x_out)

# kernel for down-sweep phase
down_ker = SourceModule("""
__global__ void down_ker(double *y, double *y_old,  int k)
{
     int j =  blockIdx.x*blockDim.x + threadIdx.x;
     
     int _2k = 1 << k;
     int _2k1 = 1 << (k+1);
     
     if (j % _2k1 == 0)
     {
         y[j + _2k - 1 ] = y_old[j + _2k1 - 1];
         y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k - 1];
     }
}
""")

down_gpu = down_ker.get_function("down_ker")

    
# implementation of down-sweep phase
def down_sweep(y):
    y = np.float64(y)
    y[-1] = 0
    y_gpu = gpuarray.to_gpu(y)
    y_old_gpu = y_gpu.copy()
    for k in reversed(range(int(np.log2(y.size)))):
        down_gpu(y_gpu, y_old_gpu, np.int32(k), block=(32,1,1), grid=(int(y.size)/32,1,1))
        y_old_gpu[:] = y_gpu[:]
    y_out = y_gpu.get()
    return(y_out)
    
   
# full implementation of work-efficient parallel prefix sum
def efficient_prefix(x):
        return(down_sweep(up_sweep(x)))



if __name__ == '__main__':
    
    
    testvec = np.random.randn(16*1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)
    
    outvec_gpu = gpuarray.empty_like(testvec_gpu)
     
    prefix_sum = np.roll(np.cumsum(testvec), 1)
    prefix_sum[0] = 0
    
    prefix_sum_gpu = efficient_prefix(testvec)
    
    print "Does our work-efficient prefix work? {}".format(np.allclose(prefix_sum_gpu, prefix_sum))
    
    
