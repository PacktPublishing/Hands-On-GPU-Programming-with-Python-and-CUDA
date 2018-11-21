from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
from timeit import timeit

SumCode='''
// we will use __inline__ to indicate to the compiler to use these as macros.
// (saves some time from calling these functions.)
__device__  void __inline__ laneid(int & id)
{
 asm("mov.u32 %0, %%laneid; " : "=r"(id)); 
}

__device__ void __inline__ split64(double val, int & lo, int & hi)
{
 asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(lo),"=r"(hi):"d"(val));
}

__device__ void __inline__ combine64(double &val, int lo, int hi)
{
 asm volatile("mov.b64 %0, {%1, %2}; ":"=d"(val):"r"(lo),"r"(hi));
}


__global__ void sum_ker(double *input, double *out) 
{

 // use PTX assembly to get lane ID
 
 int id;
 laneid(id);
 
 // vectorized memory load 

 double2 vals = *reinterpret_cast<double2*> ( &input[(blockDim.x*blockIdx.x + threadIdx.x) * 2] );
 
 double sum_val = vals.x + vals.y;

 double temp;
 
 int s1, s2;
 
 for (int i=1; i < 32; i *= 2)
 {

     
     // use PTX assembly to split
     split64(sum_val, s1, s2);
 
     // shuffle to transfer data
     s1 = __shfl_down (s1, i, 32);
     s2 = __shfl_down (s2, i, 32);
     
     
     // PTX assembly to combine
     combine64(temp, s1, s2);
     sum_val += temp;
 }

 // if we are lane 0, add to the final output
 if (id == 0)
     atomicAdd(out, sum_val);
     
}'''
 
sum_mod = SourceModule(SumCode)
sum_ker = sum_mod.get_function('sum_ker')

a = np.float64(np.random.randn(100000*2*32))
a_gpu = gpuarray.to_gpu(a)
out = gpuarray.zeros((1,), dtype=np.float64)

sum_ker(a_gpu, out, grid=(int(np.ceil(a.size/64)),1,1), block=(32,1,1))
drv.Context.synchronize()

print 'Does sum_ker produces the same value as NumPy\'s sum (according allclose)? : %s' % np.allclose(np.sum(a) , out.get()[0])

print 'Performing sum_ker / PyCUDA sum timing tests (20 each)...'

sum_ker_time = timeit('''from __main__ import sum_ker, a_gpu, out, np, drv \nsum_ker(a_gpu, out, grid=(int(np.ceil(a_gpu.size/64)),1,1), block=(32,1,1)) \ndrv.Context.synchronize()''', number=20)
pycuda_sum_time = timeit('''from __main__ import gpuarray, a_gpu, drv \ngpuarray.sum(a_gpu) \ndrv.Context.synchronize()''', number=20)

print 'sum_ker average time duration: %s, PyCUDA\'s gpuarray.sum average time duration: %s' % (sum_ker_time, pycuda_sum_time)
print '(Performance improvement of sum_ker over gpuarray.sum: %s )' % (pycuda_sum_time / sum_ker_time)
