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
    POINTER(c_bool)
]
msq_cpp.restype = None


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


@app.route('/init')
def init():
    global DATA_PATH
    global METADATA
    global DENSITY
    global SIMILARITY
    global DENSITY_NORM

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

    index = request.args.get("index")
    index = np.array(json.loads(index)["data"]).astype(np.int32)

    list_similarity = SIMILARITY[index]
    similarity_sum = np.sum(list_similarity, axis=0)
    similarity_sum /= np.max(similarity_sum)

    return jsonify(similarity_sum.tolist())




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
num_index = 2
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

threshold = 0.5
grid_info_raw = np.zeros((resolution - 1) * (resolution - 1) * 4).astype(np.bool)

grid_info = (c_bool * ((resolution - 1) * (resolution - 1) * 4))(*grid_info_raw)

def msq_run():
    msq_cpp(output_pixel_value, threshold, resolution, grid_info) 

msq_time = timeit.timeit(msq_run, number=1)

print(kde_time)
print(msq_time)






result = np.reshape(
    np.ctypeslib.as_array(output_pixel_value), (resolution, resolution)
)

# print(point_coord_raw[index_raw[0] * 2], point_coord_raw[index_raw[0] * 2 + 1])

for i in range(resolution):
    for j in range(resolution):
        print(round(result[i][j], 2), end =" ")
    print()

msq_result = np.reshape(
    np.ctypeslib.as_array(grid_info), (resolution - 1, resolution - 1, 4)
)

for i in range(resolution - 1):
    for j in range(resolution - 1 ):
        re = 0
        for k in range(4):
            re = re + msq_result[i, j, k]
        print(re, end = " ")
    print()
    

