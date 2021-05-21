// Marching squares Algorithm implementation

#include <iostream>
#include <chrono>

using namespace std;

// #ifdef __APPLE__
// #include "/usr/local/opt/libomp/include/omp.h"
// #elif __MACH__
// #include "/usr/local/opt/libomp/include/omp.h"
// #elif __linux__
// #include <omp.h>
// #endif


void _Marching_square_algorithm(float* output_pixel_info, float Threshold, int resolution, bool* grid_info){
    int point_diff[4] = {0, 1, resolution, resolution + 1};
    //#pragma omp parallel for num_threads(8) schedule(static)
    for(int i = 0; i < resolution - 1; i++)
        for(int j = 0; j < resolution - 1; j++)
            for(int k = 0; k < 4; k++)
                grid_info[(i * (resolution - 1) + j) * 4 + k] = (output_pixel_info[i * resolution + j + point_diff[k]] >= Threshold);
    return;
}
extern "C" {
    // grid_info should be initialized as 0
    void marching_square_algorithm(
        float* output_pixel_info,
        float Threshold,
        int resolution,
        bool* grid_info
    ) {
        _Marching_square_algorithm(output_pixel_info, Threshold, resolution, grid_info);
    }
}