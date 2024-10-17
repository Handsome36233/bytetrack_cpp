#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <vector>
#include <tuple>
#include <set>

// 扩展代价矩阵为方阵
std::vector<std::vector<float>> pad_cost_matrix(const std::vector<std::vector<float>>& cost_matrix);

// 匈牙利算法
std::pair<float, std::vector<std::pair<int, int>>> hungarian_algorithm(std::vector<std::vector<float>>& cost_matrix);

// 线性分配函数
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>> linear_assignment(std::vector<std::vector<float>>& cost_matrix, float thresh, int N, int M);

#endif // HUNGARIAN_H