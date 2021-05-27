'''
FLASK SERVER Libaray import
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

import json


'''
Libraries / methods for main Computation
- kde_cpp : kernel density estimation
- msq_cpp : marching squares algorithm (TODO)
'''
from ctypes import *
import numpy as np
import timeit
import pyclipper

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

msq_cpp = b_lib.marching_square_algorithm
msq_cpp.argtypes = [
    POINTER(c_float), ## output_pixel_info
    c_float,
    c_int,
    POINTER(c_bool),
    POINTER(c_float)
]
msq_cpp.restype = c_int


'''
SERVER CODE
'''

app = Flask(__name__)
CORS(app)

DATA_PATH = "./dataset/"

METADATA   = None
DENSITY    = None
SIMILARITY = None
EMB        = None
LABEL      = None
POINT_NUM  = None

DENSITY_NORM = None


def parseArgs(request):
    dataset = request.args.get('dataset', 'mnist') ## use spheres as default
    method  = request.args.get('method', 'pca')
    sample  = request.args.get('sample', '5')
    return dataset, method, sample

def normalize(positions):
    positions = np.array(positions)
    positions[:,0] = 2 * ((positions[:,0] - np.min(positions[:,0])) 
                        / (np.max(positions[:,0]) - np.min(positions[:,0]))) - 1
    positions[:,1] = 2 * ((positions[:,1] - np.min(positions[:,1])) 
                        / (np.max(positions[:,1]) - np.min(positions[:,1]))) - 1
    return positions.tolist()

def getArrayData(request, key_name):
    array_data = request.args.get(key_name)
    array_data = np.array(json.loads(array_data)["data"]).astype(np.int32)
    return array_data


@app.route('/init')
def init():
    global DATA_PATH
    global METADATA
    global DENSITY
    global SIMILARITY
    global DENSITY_NORM
    global POINT_NUM
    global EMB_1D

    dataset, method, sample = parseArgs(request)
    path = DATA_PATH + dataset + "/" + method + "/" + sample + "/"
    if not Path(path + "snn_density.json").exists():
        return "failed", 400

    metadata_file = open(path + "metadata.json")
    density_file = open(path + "snn_density.json")
    similarity_file  = open(path + "snn_similarity.json")
    emb_file = open(path + "emb.json")
    label_file = open(path + "label.json")

    METADATA   = json.load(metadata_file)
    DENSITY    = json.load(density_file)
    SIMILARITY = json.load(similarity_file)
    EMB        = normalize(json.load(emb_file))
    LABEL      = json.load(label_file)
    
    POINT_NUM  = len(LABEL)

    EMB_1D     = np.array(EMB).reshape(POINT_NUM * 2)

    density_np = np.array(DENSITY) * METADATA["max_snn_density"]
    DENSITY_NORM = (density_np - np.min(density_np))
    DENSITY_NORM = (DENSITY_NORM / np.max(DENSITY_NORM)).tolist()

    # Should change file format later
    for i, _ in enumerate(SIMILARITY):
        SIMILARITY[i] = SIMILARITY[i]["similarity"]
        SIMILARITY[i][i] = 0

    SIMILARITY = np.array(SIMILARITY)

    return jsonify({
        "density": DENSITY_NORM,
        "emb"    : EMB
    })

@app.route('/similarity')
def similarity():
    global SIMILARITY

    index = getArrayData(request, "index")

    list_similarity = SIMILARITY[index]
    similarity_sum = np.sum(list_similarity, axis=0)
    similarity_sum /= np.max(similarity_sum)

    return jsonify(similarity_sum.tolist())


@app.route('/positionupdate')
def position_update():
    global POINT_NUM
    global EMB_1D

    ## variable setting for kernel density estimation
    index_raw   = getArrayData(request, "index")
    resolution  = int(request.args.get("resolution"))


    threshold   = float(request.args.get("threshold"))

    index_num = len(index_raw)
    cur_emb = (c_float * (POINT_NUM * 2))(*EMB_1D)
    index   = (c_int * index_num)(*index_raw)
    output_pixel_value_raw = np.zeros(resolution * resolution)
    output_pixel_value = (c_float * (resolution * resolution))(*output_pixel_value_raw)
    grid_info_raw = np.zeros((resolution - 1) * (resolution - 1) * 4).astype(np.bool)
    grid_info = (c_bool * ((resolution - 1) * (resolution - 1) * 4))(*grid_info_raw)
    
    # run kde
    kde_cpp(POINT_NUM, cur_emb, index_num, index, resolution, output_pixel_value)
    msq_cpp(output_pixel_value, threshold, resolution, grid_info)

    kde_result = np.reshape(
        np.ctypeslib.as_array(output_pixel_value), (resolution, resolution)
    ).tolist()
    msq_result = np.reshape(
        np.ctypeslib.as_array(grid_info), (resolution - 1, resolution - 1, 4)
    ).tolist()

    return jsonify({
        "kde_result": kde_result,
        "msq_result": msq_result
    })

# if __name__ == '__main__':
#     app.run(debug=True)







'''
TEST CODE
'''


#### TEST ####

## KDE
resolution = 10

num_points = 50000
point_coord_raw = np.random.rand(num_points * 2).astype(np.float32) * resolution
num_index = 1
index_raw = np.random.randint(num_points, size=num_index)
bandwidth = num_index**(-1./(2+4))



output_pixel_value_raw = np.zeros(resolution * resolution)


point_coord = (c_float * (num_points * 2))(*point_coord_raw)
index = (c_int * num_index)(*index_raw)
output_pixel_value = (c_float * (resolution * resolution))(*output_pixel_value_raw)

def kde_run():
    kde_cpp(num_points, point_coord, num_index, index, resolution, output_pixel_value)

kde_time = timeit.timeit(kde_run, number=1)



## MSQ

threshold = 0.1
grid_info_raw = np.zeros((resolution + 1) * (resolution + 1) * 4).astype(np.bool)

grid_info = (c_bool * ((resolution + 1) * (resolution + 1) * 4))(*grid_info_raw)

msq_points_raw = np.zeros(resolution * resolution * 2).astype(np.float)
msq_points = (c_float * (resolution * resolution * 2))(*msq_points_raw)

msq_size = msq_cpp(output_pixel_value, threshold, resolution, grid_info, msq_points) 

print(msq_points)
print(msq_size)


msq_points_result = np.reshape(
    np.ctypeslib.as_array(msq_points)[:msq_size * 2], (msq_size, 2)
)

print(msq_points_result)


result = np.reshape(
    np.ctypeslib.as_array(output_pixel_value), (resolution, resolution)
)

# print(point_coord_raw[index_raw[0] * 2], point_coord_raw[index_raw[0] * 2 + 1])

for i in range(resolution):
    for j in range(resolution):
        print(round(result[j][i], 2), end =" ")
    print()

msq_result = np.reshape(
    np.ctypeslib.as_array(grid_info), (resolution + 1, resolution + 1, 4)
)

for i in range(resolution + 1):
    for j in range(resolution + 1 ):
        re = 0
        for k in range(4):
            re = re + msq_result[j, i, k]
        print(re, end = " ")
    print()
    
