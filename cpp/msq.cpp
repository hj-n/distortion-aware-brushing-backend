// Marching squares Algorithm implementation

#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

// #ifdef __APPLE__
// #include "/usr/local/opt/libomp/include/omp.h"
// #elif __MACH__
// #include "/usr/local/opt/libomp/include/omp.h"
// #elif __linux__
// #include <omp.h>
// #endif

int get_case(bool* grid_info, int x, int y, int resolution) { 

    int grid_x = x + 1;
    int grid_y = y + 1;

    bool ul = grid_info[(grid_x * (resolution + 1) + grid_y) * 4];
    bool ur = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 1];
    bool ll = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 2];
    bool lr = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 3];

    int result = 0;
    result += (ul == true ? 8 : 0);
    result += (ur == true ? 4 : 0);
    result += (lr == true ? 2 : 0);
    result += (ll == true ? 1 : 0);


    return result;
}

float mix(float val1, float val2, int offset) {
    return ((float) offset) +  val1 / (val1 + val2);
}

vector<vector<float> > extract_points(float* pixel_info, bool* grid_info, int resolution) {
    int point_diff[4] = {0, resolution, 1, resolution + 1};
    int x_diff[4]     = {0, 1, 0, 1};
    int y_diff[4]     = {0, 0, 1, 1};

    vector<vector<float> > points;
    
    int step_x, step_y;
    int fx, fy;

    
    for(int i = -1; i < resolution; i++)
        for (int j = -1; j < resolution; j++) {
            for (int k = 0; k < 4; k++) {
                int case_num = get_case(grid_info, i, j, resolution);
                if (case_num < 15 && case_num > 0) {
                    step_x = i; step_y = j;
                    fx = i;     fy = j;
                    goto EXTRACTION;
                }
            }
        }
    
    return points;

EXTRACTION:
    char dir = 'X';
    char last_dir;

    int runNum = 0;

    do {
        vector<float> v(2);


        float* vals = new float[4];

        for(int k = 0; k < 4; k++) {
            float val;
            int pixel_x_idx = step_x + x_diff[k];
            int pixel_y_idx = step_y + y_diff[k];
            if(pixel_x_idx == -1 || pixel_x_idx == resolution ||
                pixel_y_idx == -1 || pixel_y_idx == resolution ) val = 0;
            else val = pixel_info[pixel_x_idx * resolution + pixel_y_idx];

            vals[k] = val;
        }
        float ul_val = vals[0]; 
        float ur_val = vals[1];
        float ll_val = vals[2];
        float lr_val = vals[3];


        int grid_x = step_x + 1;
        int grid_y = step_y + 1;

        bool ul = grid_info[(grid_x * (resolution + 1) + grid_y) * 4];
        bool ur = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 1];
        bool ll = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 2];
        bool lr = grid_info[(grid_x * (resolution + 1) + grid_y) * 4 + 3];



        // refer https://en.wikipedia.org/wiki/Marching_squares for the rules 
        switch(get_case(grid_info, step_x, step_y, resolution)) {
            case 1:  v[0] = step_x;                      v[1] = mix(ul_val, ll_val, step_y);     dir = 'D'; break;
            case 2:  v[0] = mix(ll_val, lr_val, step_x); v[1] = step_y + 1;                      dir = 'R'; break;
            case 3:  v[0] = step_x;                      v[1] = mix(ul_val, ll_val, step_y);     dir = 'R'; break;
            case 4:  v[0] = step_x + 1;                  v[1] = mix(ur_val, lr_val, step_y);     dir = 'U'; break;
            
            case 6:  v[0] = mix(ul_val, ur_val, step_x); v[1] = step_y + 1;                      dir = 'U'; break;
            case 7:  v[0] = step_x;                      v[1] = mix(ul_val, ll_val, step_y);     dir = 'U'; break;
            case 8:  v[0] = mix(ul_val, ur_val, step_x); v[1] = step_y;                          dir = 'L'; break;
            case 9:  v[0] = mix(ul_val, ur_val, step_x); v[1] = step_y;                          dir = 'D'; break;

            case 11: v[0] = mix(ul_val, ur_val, step_x); v[1] = step_y;                          dir = 'R'; break;
            case 12: v[0] = step_x + 1;                  v[1] = mix(ur_val, lr_val, step_y);     dir = 'L'; break;
            case 13: v[0] = step_x + 1;                  v[1] = mix(ur_val, lr_val, step_y);     dir = 'D'; break; 
            case 14: v[0] = mix(ll_val, lr_val, step_x); v[1] = step_y;                          dir = 'L'; break;

            case 5: 
                if (last_dir == 'L') { v[0] = step_x + 1;      v[1] = mix(ur_val, lr_val, step_y);     dir = 'D'; break; }
                else /* R */   { v[0] = step_x;          v[1] = mix(ul_val, ll_val, step_y);     dir = 'U'; break; }
            
            case 10: 
                if (last_dir == 'D') { v[0] = mix(ur_val, ul_val, step_x); v[1] = step_y;              dir = 'R'; break; }
                else /* U */   { v[0] = mix(lr_val, ll_val, step_x); v[1] = step_y + 1;          dir = 'L'; break; } 
        }

        // Debug code to identify the malfunction of upper switch-case statement
        assert(dir != 'X');

        points.push_back(v);

        switch(dir) {
            case 'U': step_y--; break;
            case 'D': step_y++; break;
            case 'R': step_x++; break;
            case 'L': step_x--; break;
        } 

        last_dir = dir;
        dir = 'X';

        runNum ++;
        assert(runNum < 600);


    } while (!(step_x == fx && step_y == fy));

    return points;
}

int _Marching_square_algorithm(float* output_pixel_info, float Threshold, int resolution, bool* grid_info, float* points){
    int point_diff[4] = {0, resolution, 1, resolution + 1};
    int x_diff[4]     = {0, 1, 0, 1};
    int y_diff[4]     = {0, 0, 1, 1};
    //#pragma omp parallel for num_threads(8) schedule(static)
    for(int i = 0; i < resolution + 1; i++)
        for(int j = 0; j < resolution + 1; j++)
            for(int k = 0; k < 4; k++) {
                int val;

                int pixel_x_idx = i + x_diff[k] - 1;
                int pixel_y_idx = j + y_diff[k] - 1;
                if(pixel_x_idx == -1 || pixel_x_idx == resolution ||
                   pixel_y_idx == -1 || pixel_y_idx == resolution ) val = false;
                else val = (output_pixel_info[pixel_x_idx * resolution + pixel_y_idx] >= Threshold);

                grid_info[(i * (resolution + 1) + j) * 4 + k] = val;
            }
    
  
    
    vector<vector<float> > points_vec = extract_points(output_pixel_info, grid_info, resolution);
    int points_length = points_vec.size();

    for(int i = 0; i < points_length; i++) {
        points[i * 2]     = points_vec[i][1];
        points[i * 2 + 1] = points_vec[i][0];
    }


    return points_length;

}
extern "C" {
    // grid_info should be initialized as 0
    int marching_square_algorithm(
        float* output_pixel_info,
        float Threshold,
        int resolution,
        bool* grid_info,
        float* points
    ) { 
 
        return _Marching_square_algorithm(output_pixel_info, Threshold, resolution, grid_info, points);
    }
}
