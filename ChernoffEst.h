#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CHERNOFFEST_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CHERNOFFEST_H

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include "Parameters.h"
#include "Strategy.h"
#include "Projection.h"
#include "Env.h"

void update_single_time_estimation(const vector<int> &permutation, queue<vector<double>> &estimation_exp_list, vector<double> &estimation_exp_sum, vector<int> &m1_cnt) {
    if (estimation_exp_list.size() == m) {
        for (int i = 0; i < action_size - 1; i++) {
            if (isnan(estimation_exp_list.front()[i]) or isinf(estimation_exp_list.front()[i])) {
                continue;
            } else {
                estimation_exp_sum[i] -= estimation_exp_list.front()[i];
                m1_cnt[i] -= 1;
            }
        }
        estimation_exp_list.pop();
    }
    vector<int> new_chosen_cnt(action_size - 1, 0);
    vector<int> new_all_cnt(action_size - 1, 0);
    vector<double> new_estimation_exp(action_size - 1, 0.0);
    for (int action_index = 0; action_index < action_size - 1; action_index++) {
        int last_cnt = 0;
        int sum1[K];
        int sum2[K];
        if (permutation[0] == action_index) {
            sum1[0] = 1;
        } else {
            sum1[0] = 0;
        }
        if (permutation[0] == action_size - 1) {
            last_cnt += 1;
        }
        sum2[0] = 0;
        for (int k_1 = 1; k_1 < K; k_1++) {
            sum1[k_1] = permutation[k_1] == action_index ? sum1[k_1 - 1] + 1 : sum1[k_1 - 1];
            if (permutation[k_1] == action_size - 1) {
                last_cnt += 1;
                sum2[k_1] = sum2[k_1 - 1] + sum1[k_1];
            } else {
                sum2[k_1] = sum2[k_1 - 1];
            }
        }
        new_chosen_cnt[action_index] = sum2[K - 1];
        new_all_cnt[action_index] = last_cnt * sum1[K - 1];
        new_estimation_exp[action_index] = double(new_chosen_cnt[action_index]) / new_all_cnt[action_index];
    }
    estimation_exp_list.push(new_estimation_exp);
    for (int i = 0; i < action_size - 1; i++) {
        if (new_all_cnt[i] == 0) {
            continue;
        } else {
            m1_cnt[i] += 1;
            estimation_exp_sum[i] += new_estimation_exp[i];
        }
    }
}

vector<double> chernoff_estimate(unsigned long long t, vector<double> &estimation_exp_sum, vector<int> &m1_cnt) {
    vector<double> utility;
    for (int i = 0; i < action_size - 1; i++) {
        if (m1_cnt[i] == 0) {
            utility.push_back(1.0);
        } else {
            double avg = estimation_exp_sum[i] / m1_cnt[i];
            utility.push_back(tau * log((avg) / (1 - avg)));
        }

    }
    utility.push_back(0.0);
    DirectProj(utility, utility_abs_bound);
    return utility;
}


#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CHERNOFFEST_H
