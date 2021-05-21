#ifndef PRINTMSQ_HPP
#define PRINTMSQ_HPP

#include <iostream>
#include <random>
using namespace std;

#define Resolution 20

void makeRandomInput_msq(float* output_pixel_info){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0, 1);
    for(int i = 0; i < Resolution * Resolution; i++)
        output_pixel_info[i] = dis(gen);
}

void printInput_msq(float* output_pixel_info){
    for(int i = 0; i < Resolution; i++){
        for(int j = 0; j < Resolution; j++)
            printf("%.2f ", output_pixel_info[i*Resolution + j]);
        cout << endl;
    }
    cout << endl;
}

void printGrid_info(bool* grid_info){
    for(int i = 0; i < Resolution - 1; i++){
        for(int j = 0; j < Resolution - 1; j++){
            for(int k = 0; k < 2; k++)
                printf("%d ", grid_info[(i*(Resolution-1) + j)*4+k]);
            cout << "| ";
        }
        cout << endl;
        for(int j=0;j<Resolution-1;j++){
            for(int k = 2; k < 4; k++)
                printf("%d ", grid_info[(i*(Resolution-1) + j)*4+k]);
            cout << "| ";
        }
        cout << endl;
        for(int j=0;j<Resolution-1;j++)
            cout << "- - - ";
        cout << endl;
    }
    cout << endl;
}

#endif