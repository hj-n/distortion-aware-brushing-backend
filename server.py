## FLAST SERVER

from ctypes import *

LIB_PATH = "./lib/libbackend.so"

b_lib = CDLL(LIB_PATH) ## BACKEND LIBARAY

## kernel density estimation 
kde_cpp = b_lib.kernel_density_estimation
kde_cpp.argtypes = [
    c_int,   ## num_points
    POINTER(POINTER(c_int)), ## point_coords
    c_int,  ## num_index
    POINTER(c_int), ## index
    c_float,     ## bandwidth
    c_int,       ## resolution
]
kde_cpp.restype = POINTER(POINTER(c_double))

#### TEST ####

num_points = 1000




