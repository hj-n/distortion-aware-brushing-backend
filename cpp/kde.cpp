#include <iostream>
#include <math.h>
#include <chrono>

// if linux
// #include <omp.h>

// if macos


#ifdef __APPLE__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __MACH__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __linux__
#include <omp.h>;
#endif


#include "kde.hpp"
using namespace std;

#define Max_Resolution 100

//filter with Gaussian Kernel
void _Gaussian_filter(int resolution, float bandwidth, float point_x, float point_y, float* Pixel_info){
    float r, s = 2.0 * bandwidth * bandwidth;

    for(int x = 0; x < resolution; x++){
        for(int y = 0; y < resolution; y++){
            float dx = (float) x - point_x;
            float dy = (float) y - point_y;
            float r_sq = dx * dx + dy * dy;
            float g_value = (exp(-r_sq / s)) / (M_PI * s);
            Pixel_info[x * resolution + y] += g_value;
        }
    }
    return;
}

void _Gaussian_filter_p(int resolution, float bandwidth, float point_x, float point_y, float* Pixel_info){
    float r, s = 2.0 * bandwidth * bandwidth;


    #pragma omp parallel for num_threads(8) schedule(auto) shared(r,s)
    for(int x = 0; x < resolution; x++){
        for(int y = 0; y < resolution; y++){
            float dx = (float) x - point_x;
            float dy = (float) y - point_y;
            float r_sq = dx * dx + dy * dy;
            float g_value = (exp(-r_sq / s)) / (M_PI * s);
            Pixel_info[x * resolution + y] += g_value;
        }
    }
    return;
}

void _2D_Kernel_density_estimation(int num_point, float* point_coord, int num_index, int* index, float bandwidth, int resolution, float* output_pixel_info){
    float MAX = 0;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();


    for(int i = 0; i < num_index; i++){
        int i_index = index[i];
        _Gaussian_filter(resolution, bandwidth, point_coord[i_index * 2], point_coord[i_index * 2 + 1], output_pixel_info);
    }


    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    /////////
    begin = chrono::steady_clock::now();


    for(int i = 0; i < num_index; i++){
        int i_index = index[i];
        _Gaussian_filter_p(resolution, bandwidth, point_coord[i_index * 2], point_coord[i_index * 2 + 1], output_pixel_info);
    }


    end = chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    /////////

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
        float bandwidth, 
        int resolution,
        float* output_pixel_info
    ) {
        _2D_Kernel_density_estimation(num_point, point_coord, num_index, index, bandwidth, resolution, output_pixel_info);
    }
}