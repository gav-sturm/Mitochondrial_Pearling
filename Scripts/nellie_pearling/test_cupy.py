import cupy
from scipy import ndimage as ndi
# print(cupy.__version__)
#
# import cupy as cp
# import cupy as cp
# cp.clear_memo()
# cp.cuda.memory.set_allocator()
# # cp.cuda.memory.set_pinned_memory_allocator()
#
# import cupy as cp
# print(cp.cuda.runtime.memGetInfo())

import platform
import subprocess

def get_cupy_version():
    try:
        import cupy
        return cupy.__version__
    except ImportError:
        return "CuPy not installed"

def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        cuda_version = output.split("release ")[-1].split(",")[0]
        return cuda_version
    except:
        return "CUDA not found or not in PATH"

def get_gpu_model():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle).decode('utf-8')
    except:
        return "Unable to retrieve GPU information"

print("CuPy version:", get_cupy_version())
print("CUDA version:", get_cuda_version())
print("GPU model:", get_gpu_model())
print("Operating System:", platform.platform())
import numpy as np
import cupy as cp

# Check versions
print("NumPy version:", np.__version__)
print("CuPy version:", cp.__version__)

# Test array creation and conversion
np_array = np.array([1, 2, 3])
cp_array = cp.array(np_array)

print("NumPy array:", np_array)
print("CuPy array:", cp_array)


import numpy as np

print("NumPy version:", np.__version__)

import cupy as cp
cp.clear_memo()
x = cp.array([1, 2, 3])
print(cp.sum(x))