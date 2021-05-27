/**
 * Code for Testing and Evaluating Backend methods 
 * KDE / Marching Squares Algorithm
 */

#include "./helper/printMSQ.hpp"
#include "./helper/printKDE.hpp"
#include "msq.cpp"
#include "kde.cpp"

#define Bandwidth 3
#define Threshold 0.5

int main(){

    int resolution;
    int num_point;
    float* point_coord;
    int num_index;
    int* index;
    float* output_pixel_info;    
    bool* grid_info;


    //make Input ( 일단은 num_point = num_index )
    makeRandomInput(&resolution, &num_point, &point_coord, &num_index, &index, &output_pixel_info);

    //print Randomly maked input
    printInput(num_point, point_coord, num_index, index);

    // // print map ( resolution * resolution )
    // printpoint(num_point, point_coord, resolution);


    //run 2D_KDE
    _2D_Kernel_density_estimation(num_point, point_coord, num_index, index, Bandwidth, resolution, output_pixel_info);

    //print Pixel_info
    printPixel_info(output_pixel_info, resolution, Threshold);

    // //make Input & Print them
    // makeRandomInput_msq(output_pixel_info, resolution, grid_info);
    allocategrid(&grid_info, resolution);

    // printInput_msq(output_pixel_info, resolution);

    // _Marching_square_algorithm(output_pixel_info, Threshold, resolution, grid_info);

    printGrid_info(grid_info, resolution);

}