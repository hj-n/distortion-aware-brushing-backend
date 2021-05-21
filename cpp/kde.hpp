#ifndef KDE_HPP
#define KDE_HPP



void _2D_Kernel_density_estimation (
    int num_point, 
    float* point_coord, 
    int num_index, 
    int* index, 
    float bandwidth, 
    int resolution,
    float* output_pixel_info
);

#endif