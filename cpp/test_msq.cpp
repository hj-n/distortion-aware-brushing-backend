#include "msq.cpp"
#include "./helper/printMSQ.hpp"
#include <random>

#ifndef Resolution
#define Resolution 30
#endif

int main(){
    float output_pixel_info[Resolution*Resolution];
    float Threshold = 0.5;
    bool grid_info[(Resolution-1)*(Resolution-1)*4] = {0, };

    //make Input & Print them
    makeRandomInput_msq(output_pixel_info);
    printInput_msq(output_pixel_info);

    marching_square_algorithm(output_pixel_info, Threshold, Resolution, grid_info);

    printGrid_info(grid_info);

}