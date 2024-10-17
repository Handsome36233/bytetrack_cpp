#include "hungarian.h"
#include <algorithm>
#include <limits>
#include <tuple>
#include <set>
#include <numeric>
#include <iostream>

using namespace std;

// 扩展代价矩阵为方阵
vector<vector<float>> pad_cost_matrix(const vector<vector<float>>& cost_matrix) {
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();
    vector<vector<float>> padded_cost_matrix;

    if (n == m) {
        return cost_matrix;
    }

    if (n < m) {
        padded_cost_matrix = cost_matrix;
        vector<float> padding(m, numeric_limits<float>::max());
        for (int i = 0; i < m - n; ++i) {
            padded_cost_matrix.push_back(padding);
        }
    }

    return padded_cost_matrix;
}

// 匈牙利算法
pair<float, vector<pair<int, int>>> hungarian_algorithm(vector<vector<float>>& cost_matrix) {
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();
    vector<vector<float>> origin_cost_matrix = cost_matrix;

    // 扩展矩阵为方阵
    cost_matrix = pad_cost_matrix(cost_matrix);
    n = cost_matrix.size();
    m = cost_matrix[0].size();

    // 每行减去最小值
    for (int i = 0; i < n; ++i) {
        float min_value = *min_element(cost_matrix[i].begin(), cost_matrix[i].end());
        for (int j = 0; j < m; ++j) {
            cost_matrix[i][j] -= min_value;
        }
    }

    // 每列减去最小值
    for (int j = 0; j < m; ++j) {
        float min_value = numeric_limits<float>::max();
        for (int i = 0; i < n; ++i) {
            min_value = min(min_value, cost_matrix[i][j]);
        }
        for (int i = 0; i < n; ++i) {
            cost_matrix[i][j] -= min_value;
        }
    }

    vector<vector<bool>> star_matrix(n, vector<bool>(m, false));
    vector<vector<bool>> prime_matrix(n, vector<bool>(m, false));
    vector<bool> row_covered(n, false);
    vector<bool> col_covered(m, false);

    // 标记每一行中的第一个零
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (cost_matrix[i][j] == 0 && !row_covered[i] && !col_covered[j]) {
                star_matrix[i][j] = true;
                row_covered[i] = true;
                col_covered[j] = true;
            }
        }
    }

    fill(row_covered.begin(), row_covered.end(), false);
    fill(col_covered.begin(), col_covered.end(), false);

    // 覆盖每一列中有星号的列
    auto cover_columns_with_stars = [&]() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (star_matrix[i][j]) {
                    col_covered[j] = true;
                }
            }
        }
    };
    cover_columns_with_stars();

    auto find_zero = [&]() {
        for (int i = 0; i < n; ++i) {
            if (!row_covered[i]) {
                for (int j = 0; j < m; ++j) {
                    if (cost_matrix[i][j] == 0 && !col_covered[j]) {
                        return make_pair(i, j);
                    }
                }
            }
        }
        return make_pair(-1, -1);
    };

    auto find_star_in_row = [&](int row) {
        for (int j = 0; j < m; ++j) {
            if (star_matrix[row][j]) {
                return j;
            }
        }
        return -1;
    };

    auto find_prime_in_row = [&](int row) {
        for (int j = 0; j < m; ++j) {
            if (prime_matrix[row][j]) {
                return j;
            }
        }
        return -1;
    };

    auto augment_path = [&](vector<pair<int, int>>& path) {
        for (const auto& p : path) {
            star_matrix[p.first][p.second] = !star_matrix[p.first][p.second];
        }
    };

    auto adjust_cost_matrix = [&]() {
        float min_uncovered_value = numeric_limits<float>::max();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (!row_covered[i] && !col_covered[j]) {
                    min_uncovered_value = min(min_uncovered_value, cost_matrix[i][j]);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (row_covered[i]) {
                for (int j = 0; j < m; ++j) {
                    cost_matrix[i][j] += min_uncovered_value;
                }
            }
        }
        for (int j = 0; j < m; ++j) {
            if (!col_covered[j]) {
                for (int i = 0; i < n; ++i) {
                    cost_matrix[i][j] -= min_uncovered_value;
                }
            }
        }
    };

    while (count(col_covered.begin(), col_covered.end(), true) < min(n, m)) {
        auto [row, col] = find_zero();
        if (row == -1) { // 没有未覆盖的零，调整代价矩阵
            adjust_cost_matrix();
            tie(row, col) = find_zero();
        }

        prime_matrix[row][col] = true;
        int star_col = find_star_in_row(row);
        if (star_col == -1) {
            // 增广路径
            vector<pair<int, int>> path;
            path.emplace_back(row, col);
            while (true) {
                int star_row = -1;
                for (int r = 0; r < n; ++r) {
                    if (star_matrix[r][path.back().second]) {
                        star_row = r;
                        break;
                    }
                }
                if (star_row == -1) break;
                path.emplace_back(star_row, path.back().second);

                int prime_col = find_prime_in_row(path.back().first);
                path.emplace_back(path.back().first, prime_col);
            }

            augment_path(path);
            fill(prime_matrix.begin(), prime_matrix.end(), vector<bool>(m, false));
            fill(row_covered.begin(), row_covered.end(), false);
            fill(col_covered.begin(), col_covered.end(), false);
            cover_columns_with_stars();
        } else {
            row_covered[row] = true;
            col_covered[star_col] = false;
        }
    }

    // 计算总代价并返回匹配
    float total_cost = 0;
    vector<pair<int, int>> result;
    for (int i = 0; i < origin_cost_matrix.size(); ++i) {
        for (int j = 0; j < origin_cost_matrix[0].size(); ++j) {
            if (star_matrix[i][j]) {
                result.emplace_back(i, j);
                total_cost += origin_cost_matrix[i][j];
            }
        }
    }
    return {total_cost, result};
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) return {};

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

// 线性分配函数
tuple<vector<pair<int, int>>, vector<int>, vector<int>> linear_assignment(vector<vector<float>>& cost_matrix, float thresh, int N, int M) {
    if (N==0 || M==0) {
        vector<pair<int, int>> empty_matches;
        vector<int> unmatched_a;
        vector<int> unmatched_b;
        if (N>0) {
            unmatched_a.resize(N);
            iota(unmatched_a.begin(), unmatched_a.end(), 0);
        }
        if (M>0) {
            unmatched_b.resize(M);
            iota(unmatched_b.begin(), unmatched_b.end(), 0);
        }

        return make_tuple(empty_matches, unmatched_a, unmatched_b);
    }
    if (N > M) {
        cost_matrix = transpose(cost_matrix);
    }
    vector<vector<float>> origin_cost_matrix = cost_matrix;

    auto [total_cost, result] = hungarian_algorithm(cost_matrix);

    vector<int> x, y;
    int n = origin_cost_matrix.size();
    int m = origin_cost_matrix[0].size();

    vector<pair<int, int>> matches;
    vector<int> unmatched_a(n), unmatched_b(m);

    iota(unmatched_a.begin(), unmatched_a.end(), 0);
    iota(unmatched_b.begin(), unmatched_b.end(), 0);

    set<int> a_i, b_i;

    for (const auto& match : result) {
        int i = match.first;
        int j = match.second;
        if (origin_cost_matrix[i][j] <= thresh) {
            x.push_back(i);
            y.push_back(j);
            if (N > M) {
                matches.emplace_back(j, i);
            } else {
                matches.emplace_back(i, j);
            }
            a_i.insert(i);
            b_i.insert(j);
        }
    }

    vector<int> unmatched_a_final, unmatched_b_final;
    for (int i = 0; i < n; ++i) {
        if (a_i.find(i) == a_i.end()) {
            unmatched_a_final.push_back(i);
        }
    }
    for (int j = 0; j < m; ++j) {
        if (b_i.find(j) == b_i.end()) {
            unmatched_b_final.push_back(j);
        }
    }

    return N>M ? make_tuple(matches, unmatched_b_final, unmatched_a_final) : make_tuple(matches, unmatched_a_final, unmatched_b_final);
}
