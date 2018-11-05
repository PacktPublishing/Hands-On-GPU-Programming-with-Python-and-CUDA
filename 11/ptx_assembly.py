from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

PtxCode='''

__device__ void set_to_zero(int &x)
{
 asm("mov.s32 %0, 0;" : "=r"(x));
}

__device__ void add_floats(float &out, float in1, float  in2)
{
 asm("add.f32 %0, %1, %2 ;" : "=f"(out) : "f"(in1) , "f"(in2));
}

__device__ void plusplus(int &x)
{
 asm("add.s32 %0, %0, 1;" : "+r"(x));
}

__device__  int laneid()
{
 int id; 
 asm("mov.u32 %0, %%laneid; " : "=r"(id)); 
 return id;
}

__device__ void split64(double val, int & lo, int & hi)
{
 asm volatile("mov.b64 {%0, %1}, %2; ":"=r"(lo),"=r"(hi):"d"(val));
}

__device__ void combine64(double &val, int lo, int hi)
{
 asm volatile("mov.b64 %0, {%1, %2}; ":"=d"(val):"r"(lo),"r"(hi));
}

__global__ void ptx_test_ker() {     

 int x=123;
 
 printf("x is %d \\n", x);
 
 set_to_zero(x);
 
 printf("x is now %d \\n", x);
 
 plusplus(x);
 
 printf("x is now %d \\n", x);
 
 float f;
 
 add_floats(f, 1.11, 2.22 );
 
 printf("f is now %f  \\n", f);
 
 printf("lane ID: %d \\n", laneid() );
 
 double orig = 3.1415;

 int t1, t2;
 
 split64(orig, t1, t2);
 
 double recon;
 
 combine64(recon, t1, t2);
 
 printf("Do split64 / combine64 work? : %s \\n", (orig == recon) ? "true" : "false"); 
 
}'''

ptx_mod = SourceModule(PtxCode)
ptx_test_ker = ptx_mod.get_function('ptx_test_ker')
ptx_test_ker(grid=(1,1,1), block=(1,1,1))
