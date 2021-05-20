#include <iostream>
#include <math.h>
#include <chrono>


#ifdef __APPLE__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __MACH__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __linux__
#include <omp.h>
#endif


#include "kde.hpp"
using namespace std;


//filter with Gaussian Kernel
void _Gaussian_filter(int resolution, float bandwidth, float point_x, float point_y, float* Pixel_info){
    float r, s = 2.0 * bandwidth * bandwidth;

    for(int i = 0; i < resolution; i++){
        for(int j = 0; j < resolution; j++){
            float dy = (float) i - point_y;
            float dx = (float) j - point_x;
            float r_sq = dx * dx + dy * dy;
            float g_value = (exp(-r_sq / s)) / (M_PI * s);
            Pixel_info[i * resolution + j] += g_value;
        }
    }
    return;
}

// parallelized version of Kernel
void _Gaussian_filter_parallel(int resolution, float bandwidth, float point_x, float point_y, float* Pixel_info){
    float r, s = 2.0 * bandwidth * bandwidth;
    float pi_s = s * M_PI;

    #pragma omp parallel for num_threads(8) schedule(auto) shared(r,s)
    for(int i = 0; i < resolution; i++){
        for(int j = 0; j < resolution; j++){
            float dy = (float) i - point_y;
            float dx = (float) j - point_x;
            float r_sq = - (dx * dx + dy * dy) / s;
            float g_value = exp(r_sq) / (pi_s);
            // float g_value = exp(r_sq) / (pi_s);

            Pixel_info[i * resolution + j] += g_value;
        }
    }
    return;
}

void _2D_Kernel_density_estimation(int num_point, float* point_coord, int num_index, int* index, float bandwidth, int resolution, float* output_pixel_info){
    float MAX = 0;

    for(int i = 0; i < num_index; i++){
        int i_index = index[i];
        _Gaussian_filter_parallel(resolution, bandwidth, point_coord[i_index * 2], point_coord[i_index * 2 + 1], output_pixel_info);
    }


    for (int i = 0; i < resolution; i++)
        for (int j = 0; j < resolution; j++) 
            MAX = max(MAX, output_pixel_info[i * resolution + j]);
        

    if(MAX != 0) {
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++)
                output_pixel_info[i * resolution + j] /= MAX;
    }

    return;
}

extern "C" {

    // output_pixel_info should be initialized as 0
    // point_coord should be pre-scaled to [0, resolution]
    void kernel_density_estimation(
        int num_point, 
        float* point_coord, 
        int num_index, 
        int* index, 
        int resolution,
        float* output_pixel_info
    ) {
        float bandwidth = pow(num_index, (-1/6));
        _2D_Kernel_density_estimation(num_point, point_coord, num_index, index, bandwidth, resolution, output_pixel_info);
    }
    
}