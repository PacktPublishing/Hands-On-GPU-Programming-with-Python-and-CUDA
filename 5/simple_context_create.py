import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv

drv.init()
dev = drv.Device(0)
ctx = dev.make_context()

x = gpuarray.to_gpu(np.float32([1,2,3]))
print x.get()

ctx.pop()
