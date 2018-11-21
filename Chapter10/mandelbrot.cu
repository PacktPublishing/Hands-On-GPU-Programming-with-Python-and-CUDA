// Compile into a shared library for ctypes (required for "mandelbrot_ctypes.py")
//  Windows: "nvcc -shared -o mandelbrot.dll mandelbrot.cu"
//  Linux: "nvcc -Xcompiler -fPIC -shared -o mandelbrot.so mandelbrot.cu"

// Compile into a PTX binary (required for "mandelbrot_ptx.py" and "mandelbrot_driver.py")
//  For both Windows and Linux:  "nvcc -ptx -o mandelbrot.ptx mandelbrot.cu"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern "C" __global__ void mandelbrot_ker(float * lattice, float * mandelbrot_graph, int max_iters, float upper_bound_squared, int lattice_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( tid < lattice_size*lattice_size )
    {
        int i = tid % lattice_size;
        int j = lattice_size - 1 - (tid / lattice_size);
        
        float c_re = lattice[i];
        float c_im = lattice[j];
        
        float z_re = 0.0f;
        float z_im = 0.0f;
        
        
        
        mandelbrot_graph[tid] = 1;
        
        for (int k = 0; k < max_iters; k++)
        {
            float temp;
            
            temp = z_re*z_re - z_im*z_im + c_re;
            z_im = 2*z_re*z_im + c_im;
            z_re = temp;
            
            if ( (z_re*z_re + z_im*z_im) > upper_bound_squared )
            {
                mandelbrot_graph[tid] = 0;
                break;
            }
        
        }
        
    }
    
    return;
}

// Linux users:  remove "__declspec(dllexport)" from the line below.
extern "C" __declspec(dllexport) void launch_mandelbrot(float * lattice,  float * mandelbrot_graph, int max_iters, float upper_bound, int lattice_size)
{
    
    int num_bytes_lattice = sizeof(float) * lattice_size;
    int num_bytes_graph = sizeof(float)* lattice_size*lattice_size;
    
    float * d_lattice;
    float * d_mandelbrot_graph;
    
    cudaMalloc((float **) &d_lattice, num_bytes_lattice);
    cudaMalloc((float **) &d_mandelbrot_graph, num_bytes_graph);
    
    cudaMemcpy(d_lattice, lattice, num_bytes_lattice, cudaMemcpyHostToDevice);
    
    int grid_size = (int)  ceil(  ( (double) lattice_size*lattice_size ) / ( (double) 32 ) );
    
    mandelbrot_ker <<< grid_size, 32 >>> (d_lattice,  d_mandelbrot_graph, max_iters, upper_bound*upper_bound, lattice_size);
    
    cudaMemcpy(mandelbrot_graph, d_mandelbrot_graph, num_bytes_graph, cudaMemcpyDeviceToHost);

    
    cudaFree(d_lattice);
    cudaFree(d_mandelbrot_graph);
    
}


