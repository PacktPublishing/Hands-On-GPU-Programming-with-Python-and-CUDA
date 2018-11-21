# Note: this code is intentionally broken!!!
# (This is intended to show a case study of how to debug CUDA code
# using printf.)

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np


ker = SourceModule('''
// row-column dot-product for matrix multiplication
__device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
{

    //printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d, N is %d.\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, N);
	float val = 0;
	
	for (int k=0; k < N; k++)
	{
    
        // broken version
        val += matrix_a[ row + k*N ] * matrix_b[ col*N + k];

        //if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        //    printf("Dot-product loop: k value is %d, matrix_a value is %f, matrix_b is %f.\\n", k, matrix_a[ row + k*N ], matrix_b[ col*N + k]);
		
        // fixed version
        //val += matrix_a[ row*N + k ] * matrix_b[ col + k*N];
	}
	
	return(val);

}

// matrix multiplication kernel that is parallelized over row/column tuples.
__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{

   // broken version
   int row = blockIdx.x + threadIdx.x;
	int col = blockIdx.y + threadIdx.y;
    
    // fixed version
    //int row = blockIdx.x*blockDim.x + threadIdx.x;
    //int col = blockIdx.y*blockDim.y + threadIdx.y;
    
    //printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d.\\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col);

	
    // broken version
    output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, col, row, N);
    
    // fixed version
	//output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, row, col, N);


}
''')

matrix_ker = ker.get_function('matrix_mult_ker')

test_a = np.float32( [xrange(1,5)] * 4 )
test_b = np.float32([xrange(14,10, -1)]*4 )

output_mat = np.matmul(test_a, test_b)

test_a_gpu = gpuarray.to_gpu(test_a)
test_b_gpu = gpuarray.to_gpu(test_b)
output_mat_gpu = gpuarray.empty_like(test_a_gpu)

matrix_ker(test_a_gpu, test_b_gpu, output_mat_gpu, np.int32(4), block=(2,2,1), grid=(2,2,1))

assert( np.allclose(output_mat_gpu.get(), output_mat) )




