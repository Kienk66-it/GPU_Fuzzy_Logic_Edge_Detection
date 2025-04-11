#include "Fuzzy.h"

float Fuzzy::trapmf(float x, float a, float b, float c, float d) {
    if (x <= a || x >= d) {
        return 0.0;
    } else if (x > a && x <= b) {
        return (x - a) / (b - a);
    } else if (x > b && x <= c) {
        return 1.0;
    } else if (x > c && x < d) {
        return (d - x) / (d - c);
    }
    return 0.0;
}

float* Fuzzy::calculate_trapmf(int* change, int size, float a, float b, float c, float d) {
    float* result = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        result[i] = trapmf(change[i], a, b, c, d);
    }
    return result;
}

float Fuzzy::gaussmf(float x, float c, float sigma) {
    return exp(-0.5 * pow((x - c) / sigma, 2));
}

float* Fuzzy::calculate_gaussmf(int* change, int size, float c, float sigma) {
    float* result = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        result[i] = gaussmf(change[i], c, sigma);
    }
    return result;
}

float Fuzzy::interp_membership(int* change, float* membership, int size, float deltaP) {
    int left = 0;
    int right = size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (change[mid] == deltaP) {
            return membership[mid];  
        } else if (change[mid] < deltaP) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (left == 0 || left == size) {
        return 0.0;
    }

    int index = left - 1;
    float x0 = change[index];
    float x1 = change[index + 1];
    float y0 = membership[index];
    float y1 = membership[index + 1];

    return y0 + (deltaP - x0) * (y1 - y0) / (x1 - x0);
}

float Fuzzy::defuzz_centroid(int* x, float* mu, int size) {
    float numerator = 0.0;
    float denominator = 0.0;

    for (int i = 0; i < size; ++i) {
        numerator += x[i] * mu[i]; 
        denominator += mu[i];       
    }

    if (denominator != 0) {
        return numerator / denominator;
    } else {
        return 0.0;
    }
}

float* Fuzzy::ffmin(float value, float* array, int size){
    float* result = (float*)malloc(size * sizeof(float));

    for(int i = 0; i < size; i++){
        if(array[i] < value) result[i] = array[i];
        else result[i] = value;
    }

    return result;
}

float* Fuzzy::ffmax(float* first_arr, float* second_arr, int size){
    float* result = (float*)malloc(size * sizeof(float));

    for(int i = 0; i < size; i++){
        if(first_arr[i] < second_arr[i]) result[i] = second_arr[i];
        else result[i] = first_arr[i];
    }

    return result;
}

bool Fuzzy::all_zero(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] != 0.0f) {
            return false;
        }
    }
    return true;  // Nếu tất cả đều là 0, trả về true
}