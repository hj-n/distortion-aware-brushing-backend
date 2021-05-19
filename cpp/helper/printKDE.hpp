#ifndef PRINTKDE_HPP
#define PRINTKDE_HPP


#include <iostream>
#include <random>
using namespace std;

#define Max_Resolution 100
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"

//print all input
void printInput(int num_point, int** point_coord, int num_index, int* index){

    cout << endl << "num_point : " << num_point << endl << "point_coord : " << endl;
    for(int i=0;i<num_point;i++){
        cout << point_coord[i][0] << " " << point_coord[i][1] << endl;
    }
    cout << endl << "num_index : " << num_index << endl << "index : ";
    for(int i=0;i<num_index;i++)
        cout << index[i] << " ";
    cout << endl << endl;
}

//print point[resolution][resolution] if selected "X" else "."
void printpoint(int num_point, int** point_coord, int resolution){
    cout << "MAP : " << endl; 
    int map[Max_Resolution][Max_Resolution] = {0, };
    for(int i = 0; i < num_point; i++)
        map[point_coord[i][0]][point_coord[i][1]] = 1;
    for(int i = 0; i < resolution; i++){
        for(int j = 0; j < resolution; j++)
            if(map[i][j])
                cout << "X ";
            else
                cout << ". ";
        cout << endl;
    }
    cout << endl;
}

//print Pixel_info[resolution][resolution]
void printPixel_info(double Pixel_info[][Max_Resolution], int resolution, double Threshold){
    cout << "Pixel Info : " << endl;
    for(int i=0;i<resolution;i++){
        for(int j=0;j<resolution;j++)
            if(Pixel_info[i][j] == 0)
                printf(" .    ");
            else if(Pixel_info[i][j] > Threshold)
                printf(ANSI_COLOR_RED "%.3lf " ANSI_COLOR_RESET, Pixel_info[i][j]);
            else
                printf("%.3lf ", Pixel_info[i][j]);
        cout << endl;
    }
    cout << endl;
}

//make Random Input
void makeRandomInput(int* resolution, int* num_point, int*** point_coord, int* num_index, int** index){

    random_device rd;
    mt19937 gen(rd());

    uniform_int_distribution<int> dis(1, 100);
    //(*resolution) = dis(gen);
    (*resolution) = 30;

    uniform_int_distribution<int> dis1(0, (*resolution)*(*resolution)-1);
    //(*num_point) = dis1(gen);
    (*num_point) = 10000;
    (*point_coord) = (int**)malloc((*num_point) * sizeof(int*));
    for(int i = 0; i < (*num_point); i++){
        (*point_coord)[i] = (int*)malloc(2 * sizeof(int));
    }
    uniform_int_distribution<int> dis2(0, (*resolution)-1);
    for(int i = 0; i < (*num_point); i++){
        (*point_coord)[i][0] = dis2(gen);
        (*point_coord)[i][1] = dis2(gen);
    }

    uniform_int_distribution<int> dis3(0, (*num_point)-1);
    //(*num_index) = dis3(gen) + 1;
    (*num_index) = (*num_point);
    (*index) = (int*)malloc((*num_index)*sizeof(int));
    for(int i = 0; i <(*num_point); i++){
        //(*index)[i] = dis3(gen);
        (*index)[i] = i;
    }

}



#endif