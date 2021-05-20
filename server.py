'''
FLASK SERVER Libaray import
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path


'''
Libraries / methods for main Computation
- kde_cpp : kernel density estimation
- msq_cpp : marching squares algorithm (TODO)
'''
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
    c_int,       ## resolution
    POINTER(c_float) ## output_pixel_value
]
kde_cpp.restype = None


'''
SERVER CODE
'''

app = Flask(__name__)
CORS(app)



@app.route('/init')
def init():
    return "TEST"

if __name__ == '__main__':
    app.run()







'''
TEST CODE
'''

'''

#### TEST ####
resolution = 25

num_points = 50000
point_coord_raw = np.random.rand(num_points * 2).astype(np.float32) * resolution
num_index = 40000
index_raw = np.random.randint(num_points, size=num_index)
bandwidth = num_index**(-1./(2+4))



output_pixel_value_raw = np.zeros(resolution * resolution)


point_coord = (c_float * (num_points * 2))(*point_coord_raw)
index = (c_int * num_index)(*index_raw)
output_pixel_value = (c_float * (resolution * resolution))(*output_pixel_value_raw)

def test():
    kde_cpp(num_points, point_coord, num_index, index, resolution, output_pixel_value)

t = timeit.timeit(test, number=1)

print(t)

result = np.reshape(
    np.ctypeslib.as_array(output_pixel_value), (resolution, resolution)
)

print(point_coord_raw[index_raw[0] * 2], point_coord_raw[index_raw[0] * 2 + 1])

for i in range(resolution):
    for j in range(resolution):
        print(round(result[i][j], 2), end =" ")
    print()

    

'''