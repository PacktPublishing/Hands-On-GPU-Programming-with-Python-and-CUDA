import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# https://docs.nvidia.com/cuda/cuda-math-api/index.html

MonteCarloKernelTemplate = '''
#include <curand_kernel.h>

#define ULL  unsigned long long
#define _R(z)   ( 1.0f / (z) )
#define _P2(z)   ( (z) * (z) )

// p stands for "precision" (single or double)
__device__ inline %(p)s  f(%(p)s x)
{
     %(p)s y;
 
     %(math_function)s;

    return y;
}


extern "C" {
__global__ void monte_carlo(int iters, %(p)s lo, %(p)s hi, %(p)s * ys_out)
{
    curandState cr_state;
     
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int num_threads = blockDim.x * gridDim.x;
    
    %(p)s t_width = (hi - lo) / ( %(p)s ) num_threads;
    
    %(p)s density = ( ( %(p)s ) iters ) / t_width;
    
    %(p)s t_lo = t_width*tid + lo;
    %(p)s t_hi = t_lo + t_width;
    

	curand_init( (ULL)  clock() + (ULL) tid, (ULL) 0, (ULL) 0, &cr_state);
    
     %(p)s y, y_sum = 0.0f;
     
     
     %(p)s rand_val, x;
     for (int i=0; i < iters; i++)
     {
         rand_val = curand_uniform%(p_curand)s(&cr_state);
         
         x = t_lo + t_width * rand_val;
         
         y_sum += f(x);
     }
     
     y = y_sum / density;
     
     ys_out[tid] = y;
}

}
'''


class MonteCarloIntegrator:
    
    def __init__(self, math_function='y = sin(x)', precision='d', lo=0, hi=np.pi, samples_per_thread=10**5, num_blocks=100):
        
        self.math_function = math_function
        
        if precision in [None, 's', 'S', 'single', np.float32]:
            self.precision = 'float'
            self.numpy_precision = np.float32
            self.p_curand = ''
        elif precision in ['d','D', 'double', np.float64]:
            self.precision = 'double'
            self.numpy_precision = np.float64
            self.p_curand = '_double'
        else:
            raise Exception('precision is invalid datatype!')
            
        if (hi - lo <= 0):
            raise Exception('hi - lo <= 0!')
        else:
            self.hi = hi
            self.lo = lo
              
        MonteCarloDict = {'p' : self.precision, 'p_curand' : self.p_curand, 'math_function' : self.math_function}
        
        self.MonteCarloCode = MonteCarloKernelTemplate % MonteCarloDict
        
        self.ker = SourceModule(no_extern_c=True , options=['-w'], source=self.MonteCarloCode)
        
        self.f = self.ker.get_function('monte_carlo')
        
        self.num_blocks = num_blocks
        
        self.samples_per_thread = samples_per_thread
        
            
    def definite_integral(self, lo=None, hi=None, samples_per_thread=None, num_blocks=None):
        
        if lo is None or hi is None:
            lo = self.lo
            hi = self.hi
            
        if samples_per_thread is None:
            samples_per_thread = self.samples_per_thread
            
        if num_blocks is None:
            num_blocks = self.num_blocks
            grid = (num_blocks,1,1)
        else:
            grid = (num_blocks,1,1)
            
        block = (32,1,1)
        
        num_threads = 32*num_blocks
        
        self.ys = gpuarray.empty((num_threads,) , dtype=self.numpy_precision)
        
        self.f(np.int32(samples_per_thread), self.numpy_precision(lo), self.numpy_precision(hi), self.ys, block=block, grid=grid)
        
        self.nintegral = np.sum(self.ys.get() )
        
        return np.sum(self.nintegral)
    
    
    
if __name__ == '__main__':

    integral_tests = [('y =log(x)*_P2(sin(x))', 11.733 , 18.472, 8.9999), ('y = _R( 1 + sinh(2*x)*_P2(log(x)) )', .9, 4, .584977), ('y = (cosh(x)*sin(x))/ sqrt( pow(x,3) + _P2(sin(x)))', 1.85, 4.81,  -3.34553) ]
    
    
    for f, lo, hi, expected in integral_tests:
        mci = MonteCarloIntegrator(math_function=f, precision='d', lo=lo, hi=hi)
        print 'The Monte Carlo numerical integration of the function\n \t f: x -> %s \n \t from x = %s to x = %s is : %s ' % (f, lo, hi, mci.definite_integral())
        print 'where the expected value is : %s\n' % expected
