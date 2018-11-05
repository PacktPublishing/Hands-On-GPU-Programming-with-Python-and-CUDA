from __future__ import division
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

DynamicParallelismCode='''
__global__ void dynamic_hello_ker(int depth)
{
 printf("Hello from thread %d, recursion depth %d!\\n", threadIdx.x, depth);
 if (threadIdx.x == 0 && blockIdx.x == 0 && blockDim.x > 1)
  {
   printf("Launching a new kernel from depth %d .\\n", depth);
   printf("-----------------------------------------\\n");
   dynamic_hello_ker<<< 1, blockDim.x - 1 >>>(depth + 1);
  }
}'''

dp_mod = DynamicSourceModule(DynamicParallelismCode)

hello_ker = dp_mod.get_function('dynamic_hello_ker')

hello_ker(np.int32(0), grid=(1,1,1), block=(4,1,1))
