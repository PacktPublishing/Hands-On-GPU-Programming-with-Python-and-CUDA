from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

AtomicCode='''
__global__ void atomic_ker(int *add_out, int *max_out) 
{

 int tid = blockIdx.x*blockDim.x + threadIdx.x;
 
 // sets *add_out to 0.  Thread-safe.
 
 atomicExch(add_out, 0);
 __syncthreads();
 
 // adds "1" to *add_out for each thread.
 atomicAdd(add_out, 1);
 
 // sets max_out to the maximum value submitted across all threads.
 atomicMax(max_out, tid);

}
'''

atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function('atomic_ker')

add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)

atomic_ker(add_out, max_out, grid=(1,1,1), block=(100,1,1))
drv.Context.synchronize()

print 'Atomic operations test:'
print 'add_out: %s' % add_out.get()[0]
print 'max_out: %s' % max_out.get()[0]
