#include <iostream>
#include <math.h>

float crossProduct(float* a, float* b) {
    return a[0] * b[1] - a[1] * b[0];
}

float norm(float* a) {
    return sqrt(a[0] * a[0] + a[1] * a[1]);
}

void add_array(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void sub_array(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

void add_array_by_ratio(float* a, float* b, float ratio, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * ratio + b[i] * (1 - ratio);
    }
}


void distance(float* p, float* lp_1, float* lp_2, float* dist) {
    float AB[2], AC[2];
    sub_array(lp_1, lp_2, AB, 2);
    sub_array(lp_1, p, AC, 2);
    (*dist) = abs(crossProduct(AB, AC)) / norm(AB);
}

void find_nearest_line(float* p, float* points, int length, int* index) {
    float dist;
    distance(p, &points[2 * length - 2], &points[0], &dist);
    index[0] = length - 1;
    index[1] = 0;
    for (int i = 0; i < length - 1; i++) {
        float cur_dist;
        distance(p, &points[2 * i], &points[2 * i + 2], &cur_dist);
        if (cur_dist < dist) {
            dist = cur_dist;
            index[0] = i;
            index[1] = i + 1;
        }
    }
}
void get_intersect(float* a1, float* a2, float* b1, float* b2, float intersect[2]) {
    float d = (a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]);

    if (d == 0) {
        intersect[0] = INFINITY;
        intersect[1] = INFINITY;
        return;
    }

    float pre = crossProduct(a1, a2), post = crossProduct(b1, b2);
    intersect[0] = (pre * (b1[0] - b2[0]) - (a1[0] - a2[0]) * post) / d;
    intersect[1] = (pre * (b1[1] - b2[1]) - (a1[1] - a2[1]) * post) / d;
}
void get_new_position(float* inner_p1, float* inner_p2, float* outer_p1, float* outer_p2, float* p, float io_ratio, float new_position[2]) {
    float slope[2], p_f[2];
    sub_array(inner_p2, inner_p1, slope, 2);
    add_array(slope, p, p_f, 2);

    float left_intersect[2], right_intersect[2];
    get_intersect(inner_p1, outer_p1, p, p_f, left_intersect);
    get_intersect(inner_p2, outer_p2, p, p_f, right_intersect);

    float p_l[2], p_r[2];
    sub_array(p, left_intersect, p_l, 2);
    sub_array(p, right_intersect, p_r, 2);

    float left_dist = norm(p_l);
    float right_dist = norm(p_r);

    float lr_ratio = 1 - (left_dist / (left_dist + right_dist));

    float inner_p[2], outer_p[2];
    add_array_by_ratio(inner_p1, inner_p2, lr_ratio, inner_p, 2);
    add_array_by_ratio(outer_p1, outer_p2, lr_ratio, outer_p, 2);

    add_array_by_ratio(inner_p, outer_p, io_ratio, new_position, 2);
}