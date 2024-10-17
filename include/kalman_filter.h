#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <ctime>

using namespace std;

class KalmanFilter {
public:
    KalmanFilter();

    pair<vector<float>, vector<vector<float>>> initiate(const vector<float>& measurement);
    
    pair<vector<float>, vector<vector<float>>> predict(const vector<float>& mean, const vector<vector<float>>& covariance);

    pair<vector<vector<float>>, vector<vector<vector<float>>>> multi_predict(const vector<vector<float>>& means, const vector<vector<vector<float>>>& covariances);
    
    pair<vector<float>, vector<vector<float>>> project(const vector<float>& mean, const vector<vector<float>>& covariance);
    
    pair<vector<float>, vector<vector<float>>> update(const vector<float>& mean, const vector<vector<float>>& covariance, const vector<float>& measurement);

private:
    int ndim;  // 状态维度
    float dt;  // 时间步长
    float std_weight_position;  // 位置标准权重
    float std_weight_velocity;  // 速度标准权重
    vector<vector<float>> motion_mat;  // 运动矩阵
    vector<vector<float>> update_mat;  // 更新矩阵

    vector<float> mat_vec_mult(const vector<vector<float>>& mat, const vector<float>& vec);
    
    vector<vector<float>> mat_mult(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2);
    
    vector<vector<float>> mat_add(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2);
    
    vector<vector<float>> mat_sub(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2);
    
    vector<float> vec_sub(const vector<float>& vec1, const vector<float>& vec2);
    
    vector<float> vec_add(const vector<float>& vec1, const vector<float>& vec2);
    
    vector<vector<float>> mat_transpose(const vector<vector<float>>& mat);
    
    vector<vector<float>> mat_inverse(const vector<vector<float>>& mat);
};

#endif // KALMANFILTER_H
