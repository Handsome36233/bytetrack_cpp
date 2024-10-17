#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <ctime>
#include <kalman_filter.h>

using namespace std;


KalmanFilter::KalmanFilter() {
    ndim = 4;
    dt = 1.0;
    // 初始化运动矩阵
    motion_mat.resize(2 * ndim, vector<float>(2 * ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        motion_mat[i][ndim + i] = dt;
        motion_mat[i + ndim][i + ndim] = 1.0;  // 保持速度部分
        motion_mat[i][i] = 1.0;  // 保持位置部分
    }

    // 初始化更新矩阵
    update_mat.resize(ndim, vector<float>(2 * ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        update_mat[i][i] = 1.0;
    }

    std_weight_position = 1.0 / 20;
    std_weight_velocity = 1.0 / 160;
}

pair<vector<float>, vector<vector<float>>> KalmanFilter::initiate(const vector<float>& measurement) {
    vector<float> mean_pos(measurement.begin(), measurement.begin() + 4);
    vector<float> mean_vel(4, 0.0);
    vector<float> mean = mean_pos;
    mean.insert(mean.end(), mean_vel.begin(), mean_vel.end());

    vector<float> std_dev = {
        2 * std_weight_position * measurement[3],
        2 * std_weight_position * measurement[3],
        1e-2,
        2 * std_weight_position * measurement[3],
        10 * std_weight_velocity * measurement[3],
        10 * std_weight_velocity * measurement[3],
        1e-5,
        10 * std_weight_velocity * measurement[3]
    };

    vector<vector<float>> covariance(2 * ndim, vector<float>(2 * ndim, 0.0));
    for (int i = 0; i < 2 * ndim; ++i) {
        covariance[i][i] = std_dev[i] * std_dev[i];
    }
    return {mean, covariance};
}

pair<vector<float>, vector<vector<float>>> KalmanFilter::predict(const vector<float>& mean, const vector<vector<float>>& covariance) {
    vector<float> std_pos = {
        std_weight_position * mean[3],
        std_weight_position * mean[3],
        1e-2,
        std_weight_position * mean[3]
    };
    vector<float> std_vel = {
        std_weight_velocity * mean[3],
        std_weight_velocity * mean[3],
        1e-5,
        std_weight_velocity * mean[3]
    };

    vector<vector<float>> motion_cov(2 * ndim, vector<float>(2 * ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        motion_cov[i][i] = std_pos[i] * std_pos[i];
        motion_cov[ndim + i][ndim + i] = std_vel[i] * std_vel[i];
    }

    vector<float> new_mean = mat_vec_mult(motion_mat, mean);
    vector<vector<float>> new_covariance = mat_mult(motion_mat, covariance);
    new_covariance = mat_mult(new_covariance, mat_transpose(motion_mat));
    new_covariance = mat_add(new_covariance, motion_cov);

    return {new_mean, new_covariance};
}

pair<vector<vector<float>>, vector<vector<vector<float>>>> KalmanFilter::multi_predict(const vector<vector<float>>& means, const vector<vector<vector<float>>>& covariances) {
    size_t num_targets = means.size();
    vector<vector<float>> new_means(num_targets, vector<float>(2 * ndim, 0.0));
    vector<vector<vector<float>>> new_covariances(num_targets, vector<vector<float>>(2 * ndim, vector<float>(2 * ndim, 0.0)));

    // 进行每个目标的预测
    for (size_t i = 0; i < num_targets; ++i) {
        auto [new_mean, new_covariance] = predict(means[i], covariances[i]);
        new_means[i] = new_mean;
        new_covariances[i] = new_covariance;
    }

    return {new_means, new_covariances};
}

pair<vector<float>, vector<vector<float>>> KalmanFilter::project(const vector<float>& mean, const vector<vector<float>>& covariance) {
    vector<float> std_dev = {
        std_weight_position * mean[3],
        std_weight_position * mean[3],
        1e-1,
        std_weight_position * mean[3]
    };

    vector<vector<float>> innovation_cov(ndim, vector<float>(ndim, 0.0));
    for (int i = 0; i < ndim; ++i) {
        innovation_cov[i][i] = std_dev[i] * std_dev[i];
    }

    vector<float> projected_mean = mat_vec_mult(update_mat, mean);
    vector<vector<float>> projected_cov = mat_mult(update_mat, covariance);
    projected_cov = mat_mult(projected_cov, mat_transpose(update_mat));
    projected_cov = mat_add(projected_cov, innovation_cov);

    return {projected_mean, projected_cov};
}

pair<vector<float>, vector<vector<float>>> KalmanFilter::update(const vector<float>& mean,
                                                    const vector<vector<float>>& covariance,
                                                    const vector<float>& measurement) {
    auto [projected_mean, projected_cov] = project(mean, covariance);

    // 计算Kalman增益
    vector<vector<float>> kalman_gain = mat_mult(covariance, mat_transpose(update_mat));

    kalman_gain = mat_mult(kalman_gain, mat_inverse(projected_cov));
    vector<float> innovation = vec_sub(measurement, projected_mean);
    vector<float> new_mean = vec_add(mean, mat_vec_mult(kalman_gain, innovation));
    vector<vector<float>> new_covariance = mat_mult(kalman_gain, projected_cov);
    new_covariance = mat_mult(new_covariance, mat_transpose(kalman_gain));
    new_covariance = mat_sub(covariance, new_covariance);
    return {new_mean, new_covariance};
}


vector<float> KalmanFilter::mat_vec_mult(const vector<vector<float>>& mat, const vector<float>& vec) {
    vector<float> result(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

vector<vector<float>> KalmanFilter::mat_mult(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2) {
    assert(mat1[0].size() == mat2.size());
    vector<vector<float>> result(mat1.size(), vector<float>(mat2[0].size(), 0.0));
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat2[0].size(); ++j) {
            for (size_t k = 0; k < mat2.size(); ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

vector<vector<float>> KalmanFilter::mat_add(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2) {
    vector<vector<float>> result(mat1.size(), vector<float>(mat1[0].size(), 0.0));
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[0].size(); ++j) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    return result;
}

vector<vector<float>> KalmanFilter::mat_sub(const vector<vector<float>>& mat1, const vector<vector<float>>& mat2) {
    vector<vector<float>> result(mat1.size(), vector<float>(mat1[0].size(), 0.0));
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[0].size(); ++j)
        {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

vector<float> KalmanFilter::vec_sub(const vector<float>& vec1, const vector<float>& vec2) {
    assert(vec1.size() == vec2.size());
    vector<float> result(vec1.size(), 0.0);
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

vector<float> KalmanFilter::vec_add(const vector<float>& vec1, const vector<float>& vec2) {
    assert(vec1.size() == vec2.size());
    vector<float> result(vec1.size(), 0.0);
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

vector<vector<float>> KalmanFilter::mat_transpose(const vector<vector<float>>& mat) {
    vector<vector<float>> transposed(mat[0].size(), vector<float>(mat.size(), 0.0));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

vector<vector<float>> KalmanFilter::mat_inverse(const vector<vector<float>>& mat) {
    int n = mat.size();
    vector<vector<float>> inverse(n, vector<float>(n, 0.0));
    vector<vector<float>> augmented(n, vector<float>(2 * n, 0.0));

    // 创建增广矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i][j] = mat[i][j];
        }
        augmented[i][i + n] = 1.0;  // 右侧为单位矩阵
    }

    // 高斯消元法
    for (int i = 0; i < n; ++i) {
        // 确保对角线元素不为零
        assert(augmented[i][i] != 0);
        float diag = augmented[i][i];
        for (int j = 0; j < 2 * n; ++j) {
            augmented[i][j] /= diag;  // 归一化当前行
        }

        // 消去其他行的当前列
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                float factor = augmented[j][i];
                for (int k = 0; k < 2 * n; ++k) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    // 提取逆矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inverse[i][j] = augmented[i][j + n];
        }
    }

    return inverse;
}
