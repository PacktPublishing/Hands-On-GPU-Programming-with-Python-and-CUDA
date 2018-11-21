from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray


ShflSumCode='''
__global__ void shfl_sum_ker(int *input, int *out) {

 int temp = input[threadIdx.x];

 for (int i=1; i < 32; i *= 2)
     temp += __shfl_down (temp, i, 32);

 if (threadIdx.x == 0)
     *out = temp;

}'''

shfl_mod = SourceModule(ShflSumCode)
shfl_sum_ker = shfl_mod.get_function('shfl_sum_ker')

array_in = gpuarray.to_gpu(np.int32(range(32)))
out = gpuarray.empty((1,), dtype=np.int32)

shfl_sum_ker(array_in, out, grid=(1,1,1), block=(32,1,1))

print 'Input array: %s' % array_in.get()
print 'Summed value: %s' % out.get()[0]
print 'Does this match with Python''s sum? : %s' % (out.get()[0] == sum(array_in.get()) )
