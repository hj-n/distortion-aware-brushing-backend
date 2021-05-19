#include <iostream>
#include <math.h>
#include "kde.hpp"
using namespace std;


//filter with Gaussian Kernel
void _Gaussian_filter(int resolution, int point_x, int point_y, double Pixel_info[][Max_Resolution]){
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    for(int x = 0; x < resolution; x++){
        for(int y = 0; y < resolution; y++){
            double dx = x - point_x;
            double dy = y - point_y;
            r = sqrt(dx * dx + dy * dy);
            double g_value = (exp(-(r * r) / s)) / (M_PI * s);
            Pixel_info[x][y] += g_value;
        }
    }
    return;
}

double (*_2D_Kernel_density_estimation(int num_point, int* point_coord[2], int num_index, int index[], float bandwidth, int resolution))[Max_Resolution]{
    //size of point_coord is [num_point][2]
    //size of index is [num_index]
    static double Pixel_info[Max_Resolution][Max_Resolution] = {0, };
    double MAX = 0;

    for(int i = 0; i < num_index; i++){
        int i_index = index[i];
        _Gaussian_filter(resolution, point_coord[i_index][0], point_coord[i_index][1], Pixel_info);
    }

    for (int i = 0; i < resolution; i++)
        for (int j = 0; j < resolution; j++){
            if(Pixel_info[i][j] < bandwidth)
                Pixel_info[i][j] = 0;
            else
                MAX = max(MAX, Pixel_info[i][j]);
        }
    if(MAX != 0){
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++)
                Pixel_info[i][j] /= MAX;
    }
    return Pixel_info;
}
