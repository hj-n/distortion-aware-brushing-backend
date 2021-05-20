## FLAST SERVER

from ctypes import *

import numpy as np
import timeit

LIB_PATH = "./lib/libbackend.so"

b_lib = CDLL(LIB_PATH) ## BACKEND LIBARAY

## kernel density estimation 
kde_cpp = b_lib.kernel_density_estimation
kde_cpp.argtypes = [
    c_int,   ## num_points
    POINTER(c_float), ## point_coords
    c_int,  ## num_index
    POINTER(c_int), ## index
    c_float,     ## bandwidth
    c_int,       ## resolution
    POINTER(c_float) ## output_pixel_value
]
kde_cpp.restype = None

#### TEST ####
resolution = 100

num_points = 10000
point_coord_raw = np.random.rand(num_points * 2).astype(np.float32) * resolution
num_index = 5000
index_raw = np.random.randint(num_points, size=num_index)
bandwidth = 1

output_pixel_value_raw = np.zeros(resolution * resolution)


point_coord = (c_float * (num_points * 2))(*point_coord_raw)
index = (c_int * num_index)(*index_raw)
output_pixel_value = (c_float * (resolution * resolution))(*output_pixel_value_raw)

def test():
    kde_cpp(num_points, point_coord, num_index, index, bandwidth, resolution, output_pixel_value)

t = timeit.timeit(test, number=1)

print(t)

result = np.reshape(
    np.ctypeslib.as_array(output_pixel_value), (resolution, resolution)
)

    

