// Marching squares Algorithm implementation

#include <iostream>
#include <chrono>

using namespace std;


#ifdef __APPLE__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __MACH__
#include "/usr/local/opt/libomp/include/omp.h"
#elif __linux__
#include <omp.h>
#endif


// void _Marching_square_algorithm(float* output_pixel_info, float Threshold, int resolution, bool* grid_info){
//     int dx[4] = {0, 0, -1, -1};
//     int dy[4] = {0, -1, 0, -1};

//     #pragma omp parallel for num_threads(8) schedule(static)
//     for(int i = 0; i < resolution; i++){
//         for(int j = 0; j < resolution; j++){
//             bool is_in = (output_pixel_info[i * resolution + j] >= Threshold);
//             if(is_in){
//                 for(int k = 0; k < 4; k++){
//                     int x = i + dx[k];
//                     int y = j + dy[k];
//                     if(x < 0 || x > resolution-2 || y < 0 || y > resolution - 2) continue;
//                     grid_info[(x * (resolution - 1) + y) * 4 + k] = is_in;
//                 }
//             }
//         }
//     }
// }


void _Marching_square_algorithm_v2(float* output_pixel_info, float Threshold, int resolution, bool* grid_info){
    int point_diff[4] = {0, 1, resolution, resolution + 1};
    #pragma omp parallel for num_threads(8) schedule(static)
    for(int i = 0; i < resolution - 1; i++)
        for(int j = 0; j < resolution - 1; j++)
            for(int k = 0; k < 4; k++)
                grid_info[(i * (resolution - 1) + j) * 4 + k] = (output_pixel_info[i * resolution + j + point_diff[k]] < Threshold);
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
        // chrono::steady_clock::time_point begin = chrono::steady_clock::now();

        // _Marching_square_algorithm(output_pixel_info, Threshold, resolution, grid_info);

        // chrono::steady_clock::time_point end = chrono::steady_clock::now();

        // std::cout << "Time difference with parallel = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        
        // begin = chrono::steady_clock::now();

        _Marching_square_algorithm_v2(output_pixel_info, Threshold, resolution, grid_info);

        // end = chrono::steady_clock::now();

        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        
    }
}