#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "Fuzzy.h"

// nvcc -o Fuzzy_Edge_Detection_CPU Fuzzy_Edge_Detection_CPU.cu Fuzzy.cpp `pkg-config opencv --cflags --libs`

using namespace cv;

Fuzzy fuzz;

// Helper function for GPU membership interpolation
__device__ float interp_membership_gpu(int* x, float* mf, int size, float value) {
    if (value <= x[0]) return mf[0];
    if (value >= x[size-1]) return mf[size-1];
    
    int i;
    for(i = 1; i < size; i++) {
        if(value <= x[i]) break;
    }
    
    float slope = (mf[i] - mf[i-1]) / (float)(x[i] - x[i-1]);
    return mf[i-1] + slope * (value - x[i-1]);
}

// CUDA kernel for contrast adjustment
__global__ void contrast_adjustment_kernel(float* d_img, float* d_output, int size,
                                         float* d_Dark, float* d_Gray, float* d_Bright,
                                         float* d_Darkest, float* d_Mid, float* d_Brightest,
                                         int* d_change) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float P = d_img[idx];
        
        // Calculate membership degrees
        float P1_degree = interp_membership_gpu(d_change, d_Dark, CHANGE_SIZE, P);
        float P2_degree = interp_membership_gpu(d_change, d_Gray, CHANGE_SIZE, P);
        float P3_degree = interp_membership_gpu(d_change, d_Bright, CHANGE_SIZE, P);
        
        // Apply fuzzy rules
        float rule1[CHANGE_SIZE], rule2[CHANGE_SIZE], rule3[CHANGE_SIZE];
        float combined[CHANGE_SIZE];
        
        for(int i = 0; i < CHANGE_SIZE; i++) {
            rule1[i] = min(P1_degree, d_Darkest[i]);
            rule2[i] = min(P2_degree, d_Mid[i]);
            rule3[i] = min(P3_degree, d_Brightest[i]);
            
            combined[i] = max(max(rule1[i], rule2[i]), rule3[i]);
        }
        
        // Defuzzification using centroid method
        float numerator = 0.0f;
        float denominator = 0.0f;
        
        for(int i = 0; i < CHANGE_SIZE; i++) {
            numerator += d_change[i] * combined[i];
            denominator += combined[i];
        }
        
        d_output[idx] = (denominator > 0) ? numerator / denominator : 0;
    }
}

// GPU kernel cho tính toán gradient và edge detection
__global__ void edge_detection_kernel(float* d_img, float* d_output, float* d_gradX, float* d_gradY,
                                    int* d_change, float* d_Lower, float* d_Higher, 
                                    float* d_non_Edge, float* d_Edge,
                                    int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        int idx = row * width + col;
        float P = d_img[idx];
        
        // Lấy các điểm lân cận
        float neighbors[8] = {
            d_img[(row - 1) * width + (col - 1)],  // top-left
            d_img[(row - 1) * width + col],        // top
            d_img[(row - 1) * width + (col + 1)],  // top-right
            d_img[row * width + (col - 1)],        // left
            d_img[row * width + (col + 1)],        // right
            d_img[(row + 1) * width + (col - 1)],  // bottom-left
            d_img[(row + 1) * width + col],        // bottom
            d_img[(row + 1) * width + (col + 1)]   // bottom-right
        };

        // Tính gradient X và Y cho góc
        d_gradX[idx] = neighbors[4] - neighbors[3];  // East - West
        d_gradY[idx] = neighbors[6] - neighbors[1];  // South - North
        
        // Tính deltaP và độ membership
        float deltaP[8];
        float higher_degrees[5];
        float lower_degrees[3];
        
        for(int k = 0; k < 8; k++) {
            deltaP[k] = fabsf(neighbors[k] - P);
            if(k < 5) {
                higher_degrees[k] = interp_membership_gpu(d_change, d_Higher, CHANGE_SIZE, deltaP[k]);
            } else {
                lower_degrees[k-5] = interp_membership_gpu(d_change, d_Lower, CHANGE_SIZE, deltaP[k]);
            }
        }
        
        // Tính edge strength
        float edge_strength = 0.0f;
        for(int k = 0; k < 4; k++) {
            for(int l = k + 1; l < 5; l++) {
                float min_higher = min(higher_degrees[k], higher_degrees[l]);
                
                for(int m = 0; m < 3; m++) {
                    float edge_val = 0.0f;
                    float non_edge_val = 0.0f;
                    
                    // Simplified defuzzification
                    for(int n = 0; n < CHANGE_SIZE; n++) {
                        float edge_rule = min(min_higher, d_Edge[n]);
                        float non_edge_rule = min(lower_degrees[m], d_non_Edge[n]);
                        float combined = max(edge_rule, non_edge_rule);
                        
                        if(combined > 0) {
                            edge_val += d_change[n] * combined;
                            non_edge_val += combined;
                        }
                    }
                    
                    if(non_edge_val > 0) {
                        edge_strength = max(edge_strength, edge_val / non_edge_val);
                    }
                }
            }
        }
        
        d_output[idx] = edge_strength;
    }
}

__global__ void non_maximum_suppression_kernel(float* d_magnitude, float* d_angle, float* d_output,
                                             int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        int idx = row * width + col;
        float direction = d_angle[idx];
        float current_mag = d_magnitude[idx];
        
        // Chuẩn hóa góc về khoảng [0, 180]
        if (direction < 0) {
            direction += 180;
        }
        
        float neighbor1, neighbor2;
        
        // Xác định các điểm lân cận dựa trên hướng
        if ((0 <= direction && direction < 22.5) || (157.5 <= direction && direction < 180)) {
            neighbor1 = d_magnitude[idx + 1];
            neighbor2 = d_magnitude[idx - 1];
        }
        else if (22.5 <= direction && direction < 67.5) {
            neighbor1 = d_magnitude[idx + width - 1];
            neighbor2 = d_magnitude[idx - width + 1];
        }
        else if (67.5 <= direction && direction < 112.5) {
            neighbor1 = d_magnitude[idx + width];
            neighbor2 = d_magnitude[idx - width];
        }
        else {
            neighbor1 = d_magnitude[idx - width - 1];
            neighbor2 = d_magnitude[idx + width + 1];
        }
        
        // Kiểm tra điểm cực đại
        d_output[idx] = (current_mag >= neighbor1 && current_mag >= neighbor2) ? current_mag : 0;
    }
}

Mat contrast_adjustment(Mat img, int* change, float* Dark, float* Gray, float* Bright, 
                      float* Darkest, float* Mid, float* Brightest) {
    int h = img.rows, w = img.cols;
    int size = h * w;
    
    // Allocate device memory
    float *d_img, *d_output;
    float *d_Dark, *d_Gray, *d_Bright, *d_Darkest, *d_Mid, *d_Brightest;
    int *d_change;
    
    cudaMalloc(&d_img, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_Dark, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Gray, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Bright, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Darkest, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Mid, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Brightest, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_change, CHANGE_SIZE * sizeof(int));
    
    // Convert input image to float
    Mat float_img;
    img.convertTo(float_img, CV_32F);
    
    // Copy data to device
    cudaMemcpy(d_img, float_img.ptr<float>(0), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Dark, Dark, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Gray, Gray, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bright, Bright, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Darkest, Darkest, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mid, Mid, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Brightest, Brightest, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_change, change, CHANGE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    contrast_adjustment_kernel<<<numBlocks, blockSize>>>(d_img, d_output, size,
                                                       d_Dark, d_Gray, d_Bright,
                                                       d_Darkest, d_Mid, d_Brightest,
                                                       d_change);
    
    // Copy result back to host
    Mat resImg(h, w, CV_32F);
    cudaMemcpy(resImg.ptr<float>(0), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_img);
    cudaFree(d_output);
    cudaFree(d_Dark);
    cudaFree(d_Gray);
    cudaFree(d_Bright);
    cudaFree(d_Darkest);
    cudaFree(d_Mid);
    cudaFree(d_Brightest);
    cudaFree(d_change);
    
    // Convert back to 8-bit
    resImg.convertTo(resImg, CV_8UC1);
    return resImg;
}

Mat edge_detection(Mat img, int* change, float* Lower, float* Higher, float* non_Edge, float* Edge) {
    int h = img.rows, w = img.cols;
    int size = h * w;

    // Khởi tạo bộ nhớ trên GPU
    float *d_img, *d_output, *d_gradX, *d_gradY;
    float *d_Lower, *d_Higher, *d_non_Edge, *d_Edge;
    int *d_change;
    
    cudaMalloc(&d_img, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_gradX, size * sizeof(float));
    cudaMalloc(&d_gradY, size * sizeof(float));
    cudaMalloc(&d_Lower, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Higher, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_non_Edge, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_Edge, CHANGE_SIZE * sizeof(float));
    cudaMalloc(&d_change, CHANGE_SIZE * sizeof(int));
    
    // Chuyển dữ liệu từ CPU sang GPU
    img.convertTo(img, CV_32F);
    cudaMemcpy(d_img, img.ptr<float>(0), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lower, Lower, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Higher, Higher, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_non_Edge, non_Edge, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Edge, Edge, CHANGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_change, change, CHANGE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // Cấu hình grid và block
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, 
                  (h + blockSize.y - 1) / blockSize.y);
    
    // Chạy kernel edge detection
    edge_detection_kernel<<<gridSize, blockSize>>>(d_img, d_output, d_gradX, d_gradY,
                                                 d_change, d_Lower, d_Higher,
                                                 d_non_Edge, d_Edge, w, h);
    
    // Tính magnitude và angle
    float *d_magnitude, *d_angle;
    cudaMalloc(&d_magnitude, size * sizeof(float));
    cudaMalloc(&d_angle, size * sizeof(float));
    
    // Sử dụng thư viện cuBLAS hoặc tự implement kernel để tính magnitude và angle
    non_maximum_suppression_kernel<<<gridSize, blockSize>>>(d_output, d_angle, d_magnitude, w, h);
    
    // Copy kết quả về CPU
    Mat result(h, w, CV_32F);
    cudaMemcpy(result.ptr<float>(0), d_magnitude, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Giải phóng bộ nhớ
    cudaFree(d_img);
    cudaFree(d_output);
    cudaFree(d_gradX);
    cudaFree(d_gradY);
    cudaFree(d_Lower);
    cudaFree(d_Higher);
    cudaFree(d_non_Edge);
    cudaFree(d_Edge);
    cudaFree(d_change);
    cudaFree(d_magnitude);
    cudaFree(d_angle);
    
    // Chuyển về định dạng 8-bit
    result.convertTo(result, CV_8UC1);
    return result;
}

int main() {
    cudaFree(0);
    vector<int> change(CHANGE_SIZE);
    iota(change.begin(), change.end(), 0);

    // Initialize fuzzy sets
    float* lower_arr = fuzz.calculate_trapmf(change.data(), CHANGE_SIZE, 0, 0, 25, 50);
    float* higher_arr = fuzz.calculate_trapmf(change.data(), CHANGE_SIZE, 25, 50, 255, 255);
    float* non_edge_arr = fuzz.calculate_gaussmf(change.data(), CHANGE_SIZE, 10, 3.5);
    float* edge_arr = fuzz.calculate_gaussmf(change.data(), CHANGE_SIZE, 245, 3.5);
    float* dark_arr = fuzz.calculate_trapmf(change.data(), CHANGE_SIZE, 0, 0, 0, 25);
    float* gray_arr = fuzz.calculate_trapmf(change.data(), CHANGE_SIZE, 15, 25, 25, 35);
    float* bright_arr = fuzz.calculate_trapmf(change.data(), CHANGE_SIZE, 25, 50, 255, 255);
    float* darkest_arr = fuzz.calculate_gaussmf(change.data(), CHANGE_SIZE, 25, 3.5);
    float* mid_arr = fuzz.calculate_gaussmf(change.data(), CHANGE_SIZE, 125, 3.5);
    float* brightest_arr = fuzz.calculate_gaussmf(change.data(), CHANGE_SIZE, 225, 3.5);

    vector<float> Lower(lower_arr, lower_arr + CHANGE_SIZE);
    vector<float> Higher(higher_arr, higher_arr + CHANGE_SIZE);
    vector<float> non_Edge(non_edge_arr, non_edge_arr + CHANGE_SIZE);
    vector<float> Edge(edge_arr, edge_arr + CHANGE_SIZE);
    vector<float> Dark(dark_arr, dark_arr + CHANGE_SIZE);
    vector<float> Gray(gray_arr, gray_arr + CHANGE_SIZE);
    vector<float> Bright(bright_arr, bright_arr + CHANGE_SIZE);
    vector<float> Darkest(darkest_arr, darkest_arr + CHANGE_SIZE);
    vector<float> Mid(mid_arr, mid_arr + CHANGE_SIZE);
    vector<float> Brightest(brightest_arr, brightest_arr + CHANGE_SIZE);

    // Free the allocated memory
    free(lower_arr);
    free(higher_arr);
    free(non_edge_arr);
    free(edge_arr);
    free(dark_arr);
    free(gray_arr);
    free(bright_arr);
    free(darkest_arr);
    free(mid_arr);
    free(brightest_arr);

    Mat Image = imread("pexels-souvenirpixels-1534057.jpg");
    if(Image.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }

    Mat grayImage;
    vector<Mat> channels(3);
    split(Image, channels);
    grayImage = channels[2];

    // Create CUDA events for timing
    cudaEvent_t start_contrast, stop_contrast, start_edge, stop_edge;
    cudaEventCreate(&start_contrast);
    cudaEventCreate(&stop_contrast);
    cudaEventCreate(&start_edge);
    cudaEventCreate(&stop_edge);
    
    // Time contrast adjustment
    cudaEventRecord(start_contrast);
    Mat resImage = contrast_adjustment(grayImage, change.data(), Dark.data(), Gray.data(), Bright.data(),
                                     Darkest.data(), Mid.data(), Brightest.data());
    cudaEventRecord(stop_contrast);
    cudaEventSynchronize(stop_contrast);
    
    // Time edge detection
    cudaEventRecord(start_edge);
    resImage = edge_detection(resImage, change.data(), Lower.data(), Higher.data(), non_Edge.data(), Edge.data());
    cudaEventRecord(stop_edge);
    cudaEventSynchronize(stop_edge);
    
    // Calculate elapsed time
    float contrast_time = 0;
    float edge_time = 0;
    cudaEventElapsedTime(&contrast_time, start_contrast, stop_contrast);
    cudaEventElapsedTime(&edge_time, start_edge, stop_edge);
    
    cout << "Timing Results:" << endl;
    cout << "Contrast Adjustment: " << contrast_time << " ms" << endl;
    cout << "Edge Detection: " << edge_time << " ms" << endl;
    cout << "Total Time: " << contrast_time + edge_time << " ms" << endl;
    
    // Cleanup CUDA events
    cudaEventDestroy(start_contrast);
    cudaEventDestroy(stop_contrast);
    cudaEventDestroy(start_edge);
    cudaEventDestroy(stop_edge);

    cv::resize(Image, Image, Image.size() / 4);
    cv::resize(resImage, resImage, resImage.size() / 4);

    imshow("Original", Image);
    imshow("Edge Detection Result", resImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
