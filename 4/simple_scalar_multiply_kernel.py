import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule("""
__global__ void scalar_multiply_kernel(float *outvec, float scalar, float *vec)
{
     int i = threadIdx.x;
     outvec[i] = scalar*vec[i];
}
""")

scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")

testvec = np.random.randn(512).astype(np.float32)
testvec_gpu = gpuarray.to_gpu(testvec)
outvec_gpu = gpuarray.empty_like(testvec_gpu)

scalar_multiply_gpu( outvec_gpu, np.float32(2), testvec_gpu, block=(512,1,1), grid=(1,1,1))

print "Does our kernel work correctly? : {}".format(np.allclose(outvec_gpu.get() , 2*testvec) )
