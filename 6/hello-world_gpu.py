import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule('''
__global__ void hello_world_ker()
{
	printf("Hello world from thread %d, in block %d!\\n", threadIdx.x, blockIdx.x);
	
	__syncthreads();
	
	if(threadIdx.x == 0 && blockIdx.x == 0)
	{
		printf("-------------------------------------\\n");
		printf("This kernel was launched over a grid consisting of %d blocks,\\n", gridDim.x);
		printf("where each block has %d threads.\\n", blockDim.x);
	}
}
''')

hello_ker = ker.get_function("hello_world_ker")
hello_ker(  block=(5,1,1), grid=(2,1,1) )
