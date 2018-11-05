#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define _EPSILON 0.001
#define _ABS(x)	( x > 0.0f ? x : -x )

__host__ int allclose(float *A, float *B, int len)
{

	int returnval = 0;
	
	for (int i = 0; i < len; i++)
	{
		if ( _ABS(A[i] - B[i]) > _EPSILON )
		{
			returnval = -1;
			break;
		}
	}
	
	return(returnval);
}


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


__host__ int main()
{

	// Initialize to use first GPU.
	cudaSetDevice(0);

	// this indicates the width/height of the matrices
	int N = 4;
	
	// this will indicate how many bytes to allocate to store a test or output matrix
	int num_bytes = sizeof(float)*N*N;
	
	// input test matrix A
	float h_A[] = {	1.0,  2.0,  3.0,  4.0, \
					1.0,  2.0,  3.0,  4.0, \
					1.0,  2.0,  3.0,  4.0, \
					1.0,  2.0,  3.0,  4.0 };
					
	// input test matrix B
	float h_B[] = {	14.0,  13.0,  12.0,  11.0, \
					14.0,  13.0,  12.0,  11.0, \
					14.0,  13.0,  12.0,  11.0, \
					14.0,  13.0,  12.0,  11.0 };
	
	// expected output of A times B
	float h_AxB[] = { 140.0,  130.0,  120.0,  110.0, \
					140.0,  130.0,  120.0,  110.0, \
					140.0,  130.0,  120.0,  110.0, \
					140.0,  130.0,  120.0,  110.0 };
					
					
	// these pointers will be used for the GPU.
	// (notice how we use normal float pointers)
	float * d_A;
	float * d_B;
	float * d_output;
	
	// allocate memory for the test matrices on the GPU
	cudaMalloc((float **) &d_A, num_bytes);
	cudaMalloc((float **) &d_B, num_bytes);
	
	// copy the test matrices to the GPU
	cudaMemcpy(d_A, h_A, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, num_bytes, cudaMemcpyHostToDevice);
	
	// allocate memory for output on GPU
	cudaMalloc((float **) &d_output, num_bytes);
	
	// this will store the output from the GPU
	float * h_output;
	h_output = (float *) malloc(num_bytes);

	// setup our block and grid launch parameters with the dim3 class.
	dim3 block(2,2,1);
	dim3 grid(2,2,1);
	
	// launch our kernel
	matrix_mult_ker <<< grid, block >>> (d_A, d_B, d_output, N);
	
	// synchronize on the host, to ensure our kernel has finished executing.
	cudaDeviceSynchronize();
	
	// copy output from device to host.
	cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);

	// synchronize again.
	cudaDeviceSynchronize();
	
	// free arrays on device.
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_output);
	
	// reset the GPU.
	cudaDeviceReset();
	
	
	// Check to see if we got the expected output.
	// in both cases, remember to de-allocate h_output before returning.
	
	if (allclose(h_AxB, h_output, N*N) < 0)
	{
		printf("Error!  Output of kernel does not match expected output.\n");
		free(h_output);
		return(-1);
	}
	else
	{
		printf("Success!  Output of kernel matches expected output.\n");
		free(h_output);
		return(0);
	}


}
