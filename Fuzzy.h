#ifndef FUZZY_H
#define FUZZY_H

#include <bits/stdc++.h>

#define CHANGE_SIZE 256

using namespace std;

class Fuzzy {
public:
    float trapmf(float x, float a, float b, float c, float d);

    float* calculate_trapmf(int* change, int size, float a, float b, float c, float d);

    float gaussmf(float x, float c, float sigma);

    float* calculate_gaussmf(int* change, int size, float c, float sigma);

    float interp_membership(int* change, float* membership, int size, float deltaP);

    float defuzz_centroid(int* x, float* mu, int size);

    float* ffmin(float value, float* array, int size);

    float* ffmax(float* first_arr, float* second_arr, int size);

    bool all_zero(float* arr, int size);
};

#endif