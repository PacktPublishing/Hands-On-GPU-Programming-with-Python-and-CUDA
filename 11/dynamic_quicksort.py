from __future__ import division
import numpy as np
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
from pycuda import gpuarray
from random import shuffle

DynamicQuicksortCode='''
__device__ int partition(int * a, int lo, int hi)
{
 int i = lo;
 int pivot = a[hi];
 int temp;

 for (int k=lo; k<hi; k++)
 {
  if (a[k] < pivot)
  {
   temp = a[k];
   a[k] = a[i];
   a[i] = temp;
   i++;
  }
 }
 
 a[hi] = a[i];
 a[i] = pivot;
  
 return i;
}
  
__global__ void quicksort_ker(int *a, int lo, int hi)
{

 cudaStream_t s_left, s_right; 
 cudaStreamCreateWithFlags(&s_left, cudaStreamNonBlocking);
 cudaStreamCreateWithFlags(&s_right, cudaStreamNonBlocking);
 
 int mid = partition(a, lo, hi);
  
 if(mid - 1 - lo > 0)
   quicksort_ker<<< 1, 1, 0, s_left >>>(a, lo, mid - 1);
 if(hi - (mid + 1) > 0)
   quicksort_ker<<< 1, 1, 0, s_right >>>(a, mid + 1, hi);
    
 cudaStreamDestroy(s_left);
 cudaStreamDestroy(s_right);

}
'''

qsort_mod = DynamicSourceModule(DynamicQuicksortCode)

qsort_ker = qsort_mod.get_function('quicksort_ker')

if __name__ == '__main__':
    a = range(100)
    shuffle(a)
    
    a = np.int32(a)
    
    d_a = gpuarray.to_gpu(a)
    
    print 'Unsorted array: %s' % a
    
    qsort_ker(d_a, np.int32(0), np.int32(a.size - 1), grid=(1,1,1), block=(1,1,1))
    
    a_sorted = list(d_a.get())
    
    print 'Sorted array: %s' % a_sorted


