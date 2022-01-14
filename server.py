'''
FLASK SERVER Libaray import
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import logging

import json


'''
Libraries / methods for main Computation
- kde_cpp : kernel density estimation
- msq_cpp : marching squares algorithm 
'''
from ctypes import *
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from src.calculateMetric import clusteredMetric

import numpy as np
import timeit
import time
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
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

def get_similarity_list(index, similarity, avg_sim):

    global REAL_SIM_THRESHOLD
    list_similarity = similarity[index]
    list_similarity_sum = np.sum(list_similarity, axis=0)
    list_similarity_sum /= list_similarity.shape[0]

   

    list_similarity_sum /= avg_sim
    list_similarity_sum = np.clip(list_similarity_sum, 0, 1)

    if (len(index) == 1):
        list_similarity_sum[index[0]] = 1

    return list_similarity_sum



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


def rescalePoints(points, resolution, offset_scale):
    return np.array(points).astype(np.float32) / (0.5 * resolution * offset_scale) - 1
    

def offsetting(points, offset):
    print(offset)
    result = []
    for i, _ in enumerate(points):
        vec1 = points[i - 1] - points[i]
        vec2 = points[(i + 1) % len(points)] - points[i] 
        u_vec1 = vec1 / np.linalg.norm(vec1)
        u_vec2 = vec2 / np.linalg.norm(vec2)

        vec = - (u_vec1 + u_vec2)
        u_vec = vec / np.linalg.norm(vec)

        offsetted = points[i] + u_vec * offset

        result.append(offsetted)
    
    return np.array(result)


def distance(p, lp_1, lp_2):
    ortho_dist = np.linalg.norm(np.cross(lp_2 - lp_1, lp_1 - p)) / np.linalg.norm(lp_2 - lp_1)
    return ortho_dist


def find_nearest_line(p, points):
    dist = 1000000
    idx = [-1, -1]
    for i in range(-1, len(points) - 1):
        cur_dist = distance(p, points[i], points[i + 1])
        if cur_dist < dist:
            dist = cur_dist
            idx[0] = i 
            idx[1] = i + 1
    return idx

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


## IO_RATIO: 1 if inner, 0 if outer
def get_new_position(inner_p1, inner_p2, outer_p1, outer_p2, p, io_ratio):
    slope = inner_p2 - inner_p1
    p_f = slope + p
    left_intersect = get_intersect(inner_p1, outer_p1, p, p_f)
    right_intersect = get_intersect(inner_p2, outer_p2, p, p_f)

    left_dist = np.linalg.norm(p - left_intersect)
    right_dist = np.linalg.norm(p - right_intersect)

    lr_ratio = 1 - (left_dist / (left_dist + right_dist))
    
    inner_p = inner_p1 * lr_ratio + inner_p2 * (1 - lr_ratio)
    outer_p = outer_p1 * lr_ratio + outer_p2 * (1 - lr_ratio)

    new_position = inner_p * io_ratio + outer_p * (1 - io_ratio)

    return new_position

    
    
DATA_PATH = "./dataset/"

METADATA   = None
DENSITY    = None
SIMILARITY = None
EMB        = None
EMB_1D     = None
LABEL      = None
POINT_NUM  = None
AVERAGE_SIM = None  
ATTR       = None
UNNOMARL_RAW = None

DENSITY_NORM = None

ORIGIN_EMB    = None
ORIGIN_EMB_1D = None

REAL_SIM_THRESHOLD = 0.1 ## should sync with JS files (Brushing.js)


@app.route('/init')
def init():
    global DATA_PATH
    global METADATA
    global DENSITY
    global SIMILARITY
    global DENSITY_NORM
    global POINT_NUM
    global EMB_1D
    global EMB
    global AVERAGE_SIM
    global ORIGIN_EMB
    global ORIGIN_EMB_1D
    global LABEL
    global ATTR
    global UNNORMAL_RAW

    dataset, method, sample = parseArgs(request)
    path = DATA_PATH + dataset + "/" + method + "/" + sample + "/"
    if not Path(path + "snn_density.json").exists():
        return "failed", 400

    metadata_file = open(path + "metadata.json")
    density_file = open(path + "snn_density.json")
    similarity_file  = open(path + "snn_similarity.json")
    emb_file = open(path + "emb.json")
    origin_emb_file = open(path + "emb.json")

    if Path(path + "label.json").exists():
        label_file = open(path + "label.json")
        LABEL      = json.load(label_file)


    if Path(path + "attr.json").exists():
        attr_file = open(path + "attr.json")
        ATTR      = json.load(attr_file)
    
    if Path(path + "unnormal_raw.json").exists():
        unnormal_raw_file = open(path + "unnormal_raw.json")
        UNNORMAL_RAW = json.load(unnormal_raw_file)

    METADATA   = json.load(metadata_file)
    DENSITY    = json.load(density_file)
    SIMILARITY = json.load(similarity_file)
    EMB        = normalize(json.load(emb_file))
    ORIGIN_EMB = normalize(json.load(origin_emb_file))
    
    
    POINT_NUM  = len(EMB)

    EMB = np.array(EMB)
    EMB_1D     = (EMB).reshape(POINT_NUM * 2)
    ORIGIN_EMB = np.array(ORIGIN_EMB)
    ORIGIN_EMB_1D = (ORIGIN_EMB).reshape(POINT_NUM * 2)
    LABEL = np.array(LABEL)

    density_np = np.array(DENSITY) * METADATA["max_snn_density"]
    DENSITY_NORM = (density_np - np.min(density_np))
    DENSITY_NORM = (DENSITY_NORM / np.max(DENSITY_NORM)).tolist()

    # Should change file format later
    for i, _ in enumerate(SIMILARITY):
        SIMILARITY[i] = SIMILARITY[i]["similarity"]
        SIMILARITY[i][i] = 0

    SIMILARITY = np.array(SIMILARITY)

    ## Computing average similarity
    AVERAGE_SIM = []

    for i in range(SIMILARITY.shape[0]):
        current_sim = SIMILARITY[i]
        current_sim = current_sim[current_sim > REAL_SIM_THRESHOLD]
        AVERAGE_SIM.append(np.average(current_sim))
    
    AVERAGE_SIM = np.array(AVERAGE_SIM)

    return jsonify({
        "density" : DENSITY_NORM,
        "emb"     : EMB.tolist()
    })

@app.route('/pcp')
def pcp():

    global ATTR
    global UNNORMAL_RAW

    return jsonify({
        "attr": ATTR,
        "uraw": UNNORMAL_RAW
    })


@app.route('/updateRealSimThreshold')
def update_real_sim_threshold():
    global REAL_SIM_THRESHOLD
    global AVERAGE_SIM
    global SIMILARITY

    REAL_SIM_THRESHOLD = float(request.args.get("realSimThreshold"))

    print(REAL_SIM_THRESHOLD)

    AVERAGE_SIM = []

    for i in range(SIMILARITY.shape[0]):
        current_sim = SIMILARITY[i]
        current_sim = current_sim[current_sim > REAL_SIM_THRESHOLD]
        AVERAGE_SIM.append(np.average(current_sim))
    
    AVERAGE_SIM = np.array(AVERAGE_SIM)


    return "success"

@app.route('/simmatrix')
def sim_matrix():
    dataset, method, sample = parseArgs(request)
    path = DATA_PATH + dataset + "/" + method + "/" + sample + "/"
    
    similarity_file  = open(path + "snn_similarity.json")
    simmatrix = json.load(similarity_file)
    for i, _ in enumerate(simmatrix):
        simmatrix[i] = simmatrix[i]["similarity"]
        simmatrix[i][i] = 0

    return jsonify(simmatrix)

@app.route('/similarity')
def similarity():
    global SIMILARITY
    global AVERAGE_SIM

    index = getArrayData(request, "index")
    # real_sim_threshold = float(request.args.get("realSimThreshold")) ## not used
    list_similarity_sum = get_similarity_list(index, SIMILARITY, AVERAGE_SIM)

    return jsonify(list_similarity_sum.tolist())

@app.route('/restoreorigin')
def restore_origin():
    global ORIGIN_EMB
    global ORIGIN_EMB_1D
    global EMB
    global EMB_1D

    np.copyto(EMB, ORIGIN_EMB)
    np.copyto(EMB_1D, ORIGIN_EMB_1D)
    return "success"


@app.route('/updateorigin')
def update_origin():
    global ORIGIN_EMB
    global ORIGIN_EMB_1D
    global EMB
    global EMB_1D

    np.copyto(ORIGIN_EMB, EMB)
    np.copyto(ORIGIN_EMB_1D, EMB_1D)
    return "success"

@app.route('/restoreidx')
def restore_idx():
    global ORIGIN_EMB
    global ORIGIN_EMB_1D
    global EMB
    global EMB_1D
    global POINT_NUM
    # temp_emb = None
    # np.copyto(temp_emb, EMB)
    index = getArrayData(request, "index")

    for i in index:
        EMB[i][0] = ORIGIN_EMB[i][0]
        EMB[i][1] = ORIGIN_EMB[i][1]
    EMB_1D = (EMB).reshape(POINT_NUM * 2)
        
    return "success"

@app.route('/updateembdiff')
def update_emb_diff():
    global ORIGIN_EMB
    global ORIGIN_EMB_1D
    global EMB
    global EMB_1D
    global POINT_NUM

    index  = getArrayData(request, "index")
    x_diff = float(request.args.get("xDiff"))
    y_diff = float(request.args.get("yDiff"))

    for i in index:
        EMB[i][0] += x_diff
        EMB[i][1] -= y_diff
        ORIGIN_EMB[i][0] += x_diff
        ORIGIN_EMB[i][1] -= y_diff
    EMB_1D = (EMB).reshape(POINT_NUM * 2)
    ORIGIN_EMB_1D = (ORIGIN_EMB).reshape(POINT_NUM * 2)

    return "success"

@app.route('/calculatemetric')
def calculate_metric():
    global EMB
    global LABEL

    dataset, method, sample = parseArgs(request)
    clustered_label = getArrayData(request, "clusteredlabel")
    label_num = int(request.args.get("labelnum"))
    
    clusteredMetric(EMB, LABEL, clustered_label, label_num, dataset, method, sample)
   
    return "success"

@app.route('/positionupdate')
def position_update():
    global POINT_NUM
    global EMB_1D
    global EMB
    global SIMILARITY
    global AVERAGE_SIM
    global DENSITY

    ## variable setting for kernel density estimation
    index_raw     = getArrayData(request, "index")
    group_indices = getArrayData(request, "group")
    resolution    = int(request.args.get("resolution"))
    threshold     = float(request.args.get("threshold"))
    offset        = float(request.args.get("offset"))
    sim_threshold = float(request.args.get("simthreshold"))
    status        = request.args.get("status")

    offset_scale  = 100


    sims = get_similarity_list(index_raw, SIMILARITY, AVERAGE_SIM)

    index_num = len(index_raw)
    cur_emb = (c_float * (POINT_NUM * 2))(*((EMB_1D + 1) * (resolution * 0.5)))
    index   = (c_int * index_num)(*index_raw)
    output_pixel_value_raw = np.zeros(resolution * resolution)
    output_pixel_value = (c_float * (resolution * resolution))(*output_pixel_value_raw)
    grid_info_raw = np.zeros((resolution + 1) * (resolution + 1) * 4).astype(np.bool_)
    grid_info = (c_bool * ((resolution + 1) * (resolution + 1) * 4))(*grid_info_raw)
    
    # Run KDE
    kde_cpp(POINT_NUM, cur_emb, index_num, index, resolution, output_pixel_value)

    contour_raw = np.zeros(resolution * resolution * 2).astype(np.float64)
    contour = (c_float * (resolution * resolution * 2))(*contour_raw)

    ## Run MSQ
    c_size = msq_cpp(output_pixel_value, threshold, resolution, grid_info, contour) 
    
    contour_result = np.reshape(
        np.ctypeslib.as_array(contour)[:c_size * 2] * offset_scale, (c_size, 2)
    )

    ## CONVEX HULL to get smmoth / convex contour
    contour_hull = ConvexHull(contour_result)
    contour_result = contour_result[contour_hull.vertices]

    ## Offsetting
    contour_offsetted = offsetting(contour_result, offset_scale * offset)
    contour_faraway = offsetting(contour_result, offset_scale * offset * offset)

    contour_result = rescalePoints(contour_result, resolution, offset_scale)
    contour_offsetted = rescalePoints(contour_offsetted, resolution, offset_scale)
    contour_faraway = rescalePoints(contour_faraway, resolution, offset_scale)


    ## Points containment test 
    ### Current implementation: naive approach (using scipy)
    ### Will be accelerated if further performance gain is required
    contour_hull = Delaunay(contour_result)
    contour_offsetted_hull = Delaunay(contour_offsetted)

    inside_contour = contour_hull.find_simplex(EMB) >= 0
    inside_contour_offsetted = contour_offsetted_hull.find_simplex(EMB) >= 0
    
    is_considering = np.zeros(len(EMB))
    for idx in index_raw:
        is_considering[idx] = 1

    is_in_group = np.zeros(len(EMB))

    
    for idx in group_indices:
        is_in_group[idx] = 1

    is_in_real_group = np.zeros(len(EMB))
    if (status == "initiate"):
        max_density_idx = -1
        max_density = -1
        for idx in index_raw:
            if (DENSITY[idx] > max_density):
                max_density = DENSITY[idx]
                max_density_idx = idx
        is_in_real_group[max_density_idx] = 1
        sumsum = 0
        for idx in index_raw:
            print(1)
            if (SIMILARITY[idx, max_density_idx] / AVERAGE_SIM[max_density_idx] > sim_threshold):
                
                is_in_real_group[idx] = 1
                # sumsum += 1
        # print(sumsum)



    # is_in_real_group = np.zeros(len(EMB))
    # max_density_idx = -1
    # max_density = -1
    # for idx in group_indices:
    # #     if (DENSITY[idx] > max_density):
    # #         max_density = DENSITY[idx]
    # #         max_density_idx = idx
    # # is_in_real_group[max_density_idx] = 1
    # # for idx in group_indices:
    # #     if (SIMILARITY[idx, max_density_idx] > sim_threshold):
    # #         is_in_real_group[idx] = 1
    #     is_in_group[idx] = 1

    points_from_outside = []

    
    ## Repositioning
    ### SHOULD BE ACCELEARATED
    new_positions = []
    for (i, p) in enumerate(EMB):
        if is_in_group[i] == 1: ## if in groups
            if not inside_contour[i]:
                indices = find_nearest_line(p, contour_offsetted)
                new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                           contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                           p, 1.1)
                new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
            # elif sims[i] < sim_threshold:
            #         indices = find_nearest_line(p, contour_result)
            #         new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
            #                                    contour_offsetted[indices[0]], contour_offsetted[indices[1]],
            #                                    p, sims[i])
            #         new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
        elif is_considering[i] == 1:   ## if mousehovering
            if status!="initiate":
                if sims[i] < sim_threshold:
            # if is_in_group[i] == 0 and status=="initiate":
                    indices = find_nearest_line(p, contour_result)
                    new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                                contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                                p, sims[i])
                    new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
            else:
                if is_in_real_group[i] == 0:
                    indices = find_nearest_line(p, contour_result)
                    new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                                contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                                p, sims[i])
                    new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
        else: ## remaining points
            if sims[i] <= 0:
                sims[i] = - 0.7
            if inside_contour[i]:
                if sims[i] < sim_threshold:
                    indices = find_nearest_line(p, contour_result)
                    new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                               contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                               p, sims[i])
                    new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
            elif inside_contour_offsetted[i]:
                curr_sim = 1 if sims[i] > 1 else sims[i]
                indices = find_nearest_line(p, contour_offsetted)
                new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                            contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                            p, curr_sim)
                new_positions.append([i, float(new_pos[0]), float(new_pos[1])])
            else:
                if sims[i] >= sim_threshold:
                    curr_sim = 1 if sims[i] > 1 else sims[i]
                    indices = find_nearest_line(p, contour_faraway)
                    new_pos = get_new_position(contour_result[indices[0]]   , contour_result[indices[1]], 
                                               contour_offsetted[indices[0]], contour_offsetted[indices[1]],
                                               p, curr_sim)
                    new_positions.append([i, float(new_pos[0]), float(new_pos[1])])

                    points_from_outside.append(i)

    
    for datum in new_positions:
        EMB[datum[0]][0] = datum[1]
        EMB[datum[0]][1] = datum[2]

    EMB_1D = (EMB).reshape(POINT_NUM * 2)

    return jsonify({
        "contour": contour_result.tolist(),
        "contour_offsetted": contour_offsetted.tolist(),
        "new_positions": new_positions,
        "points_from_outside": points_from_outside
    })

if __name__ == '__main__':
    app.run(debug=True)








'''
TEST CODE


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
    
'''