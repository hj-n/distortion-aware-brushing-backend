/**
 * Code for Testing and Evaluationg Backend methods 
 * KDE / Marching Squares Algorithm
 */

#include "./helper/printKDE.hpp"
#include "kde.hpp"

#define Bandwidth 0.0001


int main(){

    int resolution;
    int num_point;
    int** point_coord;
    int num_index;
    int* index;


    //make Input ( 일단은 num_point = num_index )
    makeRandomInput(&resolution, &num_point, &point_coord, &num_index, &index);

    // //print Randomly maked input
    // printInput(num_point, point_coord, num_index, index);
    //print map ( resolution * resolution )
    printpoint(num_point, point_coord, resolution);


    //sizeof Pixel_info : [Max_Resolution][Max_Resolution]
    double (*Pixel_info)[Max_Resolution] ;
    //run 2D_KDE
    Pixel_info = _2D_Kernel_density_estimation(num_point, point_coord, num_index, index, Bandwidth, resolution);

    
    //print Pixel_info
    printPixel_info(Pixel_info, resolution, 0.4); // Threshold = 0.4
    
    return 0;
}