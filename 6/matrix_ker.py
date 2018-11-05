# This program is the "fixed" version of broken_matrix_ker.py

# This is to be used for an exercise where this code is translated to
# a pure CUDA-C version.

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np


ker = SourceModule('''
// row-column dot-product for matrix multiplication
__device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
{
	float val = 0;
	
	for (int k=0; k < N; k++)
	{
        val += matrix_a[ row*N + k ] * matrix_b[ col + k*N];
	}
	
	return(val);

}

// matrix multiplication kernel that is parallelized over row/column tuples.
__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

	output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, row, col, N);

}
''')

matrix_ker = ker.get_function('matrix_mult_ker')

test_a = np.float32([xrange(1,5)] * 4)
test_b = np.float32([xrange(14,10, -1)]*4 )

output_mat = np.matmul(test_a, test_b)

test_a_gpu = gpuarray.to_gpu(test_a)
test_b_gpu = gpuarray.to_gpu(test_b)
output_mat_gpu = gpuarray.empty_like(test_a_gpu)

matrix_ker(test_a_gpu, test_b_gpu, output_mat_gpu, np.int32(4), block=(2,2,1), grid=(2,2,1))

assert(np.allclose(output_mat_gpu.get(), output_mat) )




