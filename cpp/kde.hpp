#ifndef KDE_HPP
#define KDE_HPP


#define Max_Resolution 100

double (*_2D_Kernel_density_estimation(
    int num_point, 
    int* point_coord[2], 
    int num_index, 
    int index[], 
    float bandwidth, 
    int resolution
))[Max_Resolution];

#endif