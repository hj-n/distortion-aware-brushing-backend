#ifndef PRINTMSQ_HPP
#define PRINTMSQ_HPP

#include <iostream>
#include <random>
using namespace std;


void makeRandomInput_msq(float** output_pixel_info, int resolution, bool** grid_info){
    (*output_pixel_info) = (float*)malloc(resolution * resolution * sizeof(float));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0, 1);
    for(int i = 0; i < resolution * resolution; i++)
        (*output_pixel_info[i]) = dis(gen);

    (*grid_info) = (bool*)malloc((resolution - 1) * (resolution - 1) * 4 * sizeof(bool));
    memset((*grid_info), false, (resolution - 1) * (resolution - 1) * 4);
}

void allocategrid(bool** grid_info, int resolution){
    (*grid_info) = (bool*)malloc((resolution - 1) * (resolution - 1) * 4 * sizeof(bool));
    memset((*grid_info), false, (resolution - 1) * (resolution - 1) * 4);
}

void printInput_msq(float* output_pixel_info, int resolution){
    for(int i = 0; i < resolution; i++){
        for(int j = 0; j < resolution; j++)
            printf("%.2f ", output_pixel_info[i*resolution + j]);
        cout << endl;
    }
    cout << endl;
}

void printGrid_info(bool* grid_info, int resolution){
    for(int i = 0; i < resolution - 1; i++){
        for(int j = 0; j < resolution - 1; j++){
            for(int k = 0; k < 2; k++)
                printf("%d ", grid_info[(i*(resolution-1) + j)*4+k]);
            cout << "| ";
        }
        cout << endl;
        for(int j=0;j<resolution-1;j++){
            for(int k = 2; k < 4; k++)
                printf("%d ", grid_info[(i*(resolution-1) + j)*4+k]);
            cout << "| ";
        }
        cout << endl;
        for(int j=0;j<resolution-1;j++)
            cout << "- - - ";
        cout << endl;
    }
    cout << endl;
}

#endif