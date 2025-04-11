#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Fuzzy.h"

// nvcc -o Fuzzy_Edge_Detection_CPU Fuzzy_Edge_Detection_CPU.cu Fuzzy.cpp `pkg-config opencv --cflags --libs`

using namespace cv;

Fuzzy fuzz;

// Hàm Non-Maximum Suppression (NMS)
cv::Mat non_maximum_suppression(const cv::Mat &magnitude, const cv::Mat &angle) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;
    cv::Mat output = cv::Mat::zeros(rows, cols, CV_32F);

    // Tạo bản sao của ảnh góc để làm việc
    cv::Mat angle_copy = angle.clone();

    // Điều chỉnh góc về phạm vi [0, 180]
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (angle_copy.at<float>(i, j) < 0) {
                angle_copy.at<float>(i, j) += 180;
            }
        }
    }

    // Áp dụng Non-Maximum Suppression
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float direction = angle_copy.at<float>(i, j);
            float current_mag = magnitude.at<float>(i, j);

            float neighbor1, neighbor2;

            // Xác định hướng theo angle
            if ((0 <= direction && direction < 22.5) || (157.5 <= direction && direction < 180)) {
                // Hướng ngang (0° hoặc 180°)
                neighbor1 = magnitude.at<float>(i, j + 1);
                neighbor2 = magnitude.at<float>(i, j - 1);
            } else if (22.5 <= direction && direction < 67.5) {
                // Hướng chéo (+45°)
                neighbor1 = magnitude.at<float>(i + 1, j - 1);
                neighbor2 = magnitude.at<float>(i - 1, j + 1);
            } else if (67.5 <= direction && direction < 112.5) {
                // Hướng dọc (90°)
                neighbor1 = magnitude.at<float>(i + 1, j);
                neighbor2 = magnitude.at<float>(i - 1, j);
            } else {
                // Hướng chéo (-45°)
                neighbor1 = magnitude.at<float>(i - 1, j - 1);
                neighbor2 = magnitude.at<float>(i + 1, j + 1);
            }

            // Kiểm tra xem giá trị hiện tại có phải là điểm cực đại trong hướng không
            if (current_mag >= neighbor1 && current_mag >= neighbor2) {
                output.at<float>(i, j) = current_mag;
            } else {
                output.at<float>(i, j) = 0;
            }
        }
    }

    return output;
}

int Min(int a, int b) {
    return (a < b) ? a : b;
}

int Max(int a, int b) {
    return (a > b) ? a : b;
}

Mat contrast_adjustment(Mat img, int* change, float* Dark, float* Gray, float* Bright, 
                        float* Darkest, float* Mid, float* Brightest){
    int h = img.rows, w = img.cols;
    int Size = h * w;
    float P_output;

    Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);

    float* clone_resImg = resImg.ptr<float>(0);

    for(int i = 0; i < Size; i++){
        float P = clone_resImg[i];

        float P1_degree = fuzz.interp_membership(change, Dark, CHANGE_SIZE, P);
        float P2_degree = fuzz.interp_membership(change, Gray, CHANGE_SIZE, P);
        float P3_degree = fuzz.interp_membership(change, Bright, CHANGE_SIZE, P);

        float* rule1_degree = fuzz.ffmin(P1_degree, Darkest, CHANGE_SIZE);
        float* rule2_degree = fuzz.ffmin(P2_degree, Mid, CHANGE_SIZE);
        float* rule3_degree = fuzz.ffmin(P3_degree, Brightest, CHANGE_SIZE);

        float* combined_degree = fuzz.ffmax(rule1_degree, rule2_degree, CHANGE_SIZE);
        combined_degree = fuzz.ffmax(combined_degree, rule3_degree, CHANGE_SIZE);

        if (fuzz.all_zero(combined_degree, CHANGE_SIZE)) {
            P_output = 0;
        }
        else{
            P_output = fuzz.defuzz_centroid(change, combined_degree, CHANGE_SIZE);
        }

        clone_resImg[i] = P_output;
        
        free(rule1_degree);
        free(rule2_degree);
        free(rule3_degree);
        free(combined_degree);
    }

    resImg.convertTo(resImg, CV_8UC1);
    return resImg;
}

Mat edge_detection(Mat img, int* change, float* Lower, float* Higher, float* non_Edge, float* Edge) {
    int h = img.rows, w = img.cols;
    Mat resImg = Mat::zeros(h, w, CV_32F);
    Mat gradX = Mat::zeros(h, w, CV_32F);
    Mat gradY = Mat::zeros(h, w, CV_32F);
    img.convertTo(img, CV_32F);

    float* clone_img = img.ptr<float>(0);
    float* clone_resImg = resImg.ptr<float>(0);
    float* gx = gradX.ptr<float>(0);
    float* gy = gradY.ptr<float>(0);

    // Edge detection with gradient calculation
    for(int i = 1; i < h - 1; i++) {
        for(int j = 1; j < w - 1; j++) {
            float P = clone_img[i * w + j];
            vector<float> neighbors = {
                clone_img[(i - 1) * w + (j - 1)], clone_img[(i - 1) * w + j],
                clone_img[(i - 1) * w + (j + 1)], clone_img[i * w + (j - 1)],
                clone_img[i * w + (j + 1)], clone_img[(i + 1) * w + (j - 1)],
                clone_img[(i + 1) * w + j], clone_img[(i + 1) * w + (j + 1)]
            };
            
            vector<float> deltaP;
            vector<float> higher_degrees;
            vector<float> lower_degrees;
            
            for(int k = 0; k < 8; k++) {
                deltaP.push_back(abs((int)neighbors[k] - (int)P));
                if(k < 5) {
                    higher_degrees.push_back(fuzz.interp_membership(change, Higher, CHANGE_SIZE, deltaP[k]));
                } else {
                    lower_degrees.push_back(fuzz.interp_membership(change, Lower, CHANGE_SIZE, deltaP[k]));
                }
            }

            // Calculate gradients for angle computation
            gx[i * w + j] = neighbors[4] - neighbors[3]; // East - West
            gy[i * w + j] = neighbors[6] - neighbors[1]; // South - North
            
            // Simplified rule evaluation
            float edge_strength = 0;
            for(int k = 0; k < 4; k++) {
                for(int l = k + 1; l < 5; l++) {
                    float* edge_rule = fuzz.ffmin(Min(higher_degrees[k], higher_degrees[l]), Edge, CHANGE_SIZE);
                    for(auto lower_deg : lower_degrees) {
                        float* non_edge_rule = fuzz.ffmin(lower_deg, non_Edge, CHANGE_SIZE);
                        float* combined = fuzz.ffmax(edge_rule, non_edge_rule, CHANGE_SIZE);
                        
                        if (!fuzz.all_zero(combined, CHANGE_SIZE)) {
                            edge_strength = max(edge_strength, fuzz.defuzz_centroid(change, combined, CHANGE_SIZE));
                        }
                        
                        free(non_edge_rule);
                        free(combined);
                    }
                    free(edge_rule);
                }
            }
            
            clone_resImg[i * w + j] = edge_strength;
        }
    }

    // Calculate magnitude and angle
    Mat magnitude, angle;
    cartToPolar(gradX, gradY, magnitude, angle, true);
    
    // Apply non-maximum suppression
    Mat suppressed = non_maximum_suppression(resImg, angle);
    
    suppressed.convertTo(suppressed, CV_8UC1);
    return suppressed;
}

int main() {
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

    Mat Image = imread("Screenshot from 2024-12-15 11-14-35.png");
    if(Image.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }

    Mat grayImage;
    vector<Mat> channels(3);
    split(Image, channels);
    grayImage = channels[2];

    Mat resImage = contrast_adjustment(grayImage, change.data(), Dark.data(), Gray.data(), Bright.data(),
                                     Darkest.data(), Mid.data(), Brightest.data());
    resImage = edge_detection(resImage, change.data(), Lower.data(), Higher.data(), non_Edge.data(), Edge.data());

    imshow("Original", Image);
    imshow("Edge Detection Result", resImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
} 