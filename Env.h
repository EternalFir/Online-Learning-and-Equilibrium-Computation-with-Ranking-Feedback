#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_ENV_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_ENV_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include <random>

#include <chrono>

#include "Parameters.h"
#include "Strategy.h"
#include "Projection.h"

using namespace std;

vector<double> GetInitUtility(const int size, const double abs_bound, mt19937 &gen) {
    vector<double> utility;
    uniform_real_distribution<double> dist(-abs_bound, abs_bound);
    for (int i = 0; i < size; i++) {
        utility.push_back(dist(gen));
    }
    UtilityZero(utility);
    DirectProj(utility, abs_bound);
    return utility;
}

void UpdateUtility(vector<double> &utility, vector<pair<double, double>> &utility_range, double single_step_change, mt19937 &gen) {
    uniform_real_distribution<double> dist(-single_step_change, single_step_change);
    vector<double> origin_change(action_size, 0.0);
    for (int i = 0; i < action_size; i++) {
        origin_change[i] = dist(gen) * 0.5 * (utility_range[i].second - utility_range[i].first);
    }
    double l = 0.1 / double(action_size), r = 1e9, mid;
    int round = 0;
    double error = 10000;
    vector<double> new_utility(action_size, 0.0);
    while (r - l > 1e-6) {
        mid = (l + r) * 0.5;
        for (int i = 0; i < action_size; i++) {
            new_utility[i] = utility[i] + origin_change[i] * mid;
        }
        UtilityZero(new_utility);
        DirectProj(new_utility, utility_range);
        double actual_change = 0.0;
        for (int i = 0; i < action_size; i++) {
            actual_change += (new_utility[i] - utility[i]) * (new_utility[i] - utility[i]);
        }
        if (sqrt(actual_change) < single_step_change) {
            l = mid;
        } else {
            r = mid;
        }
        round++;
        error = abs(actual_change - single_step_change);
    }
    for (int i = 0; i < action_size; i++) {
        error += (utility[i] - new_utility[i]) * (utility[i] - new_utility[i]);
        utility[i] = new_utility[i];
    }
    return;
}

vector<int> SamplePermutationSingle(vector<double> &utility, vector<int> sampled_actions, mt19937 &gen) {
    vector<int> permutation;
    if (utility.size() != action_size) {
        throw string("invalid utility size in SamplePermutationSingle.");
    }
    if (sampled_actions.size() != K) {
        throw string("invalid sampled_actions size in SamplePermutationSingle.");
    }
    vector<double> exp_values(K, 0.0);
    vector<bool> used(K, false);
    double sum_exp = 0.0;
    for (int k_1 = 0; k_1 < K; k_1++) {
        exp_values[k_1] = exp(utility[sampled_actions[k_1]] / tau);
        sum_exp += exp_values[k_1];
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double prob;
    for (int k_1 = 0; k_1 < K; k_1++) {
        prob = dist(gen) * sum_exp;
        bool flg = false;
        for (int k_2 = 0; k_2 < K; k_2++)
            if (!used[k_2]) {
                prob -= exp_values[k_2];
                if (prob < 1e-8) {
                    flg = true;
                    permutation.push_back(sampled_actions[k_2]);
                    sum_exp -= exp_values[k_2];
                    exp_values[k_2] = 0.0;
                    used[k_2] = true;
                    break;
                }
            }
        if (!flg) {
            for (int k_2 = K - 1; k_2 >= 0; k_2--)
                if (!used[k_2]) {
                    permutation.push_back(sampled_actions[k_2]);
                    sum_exp -= exp_values[k_2];
                    exp_values[k_2] = 0.0;
                    used[k_2] = true;
                }
        }
    }
    for (int k_1 = 0; k_1 < K; k_1++) {
        if (exp_values[k_1] > 1e-6) {
            throw string("error in SamplePermutationSingle.");
        }
    }
    return permutation;
}


vector<int> SamplePermutationSingleBandit(vector<double> &sampled_utility, vector<int> sampled_actions, mt19937 &gen) {
    vector<int> permutation;
    if (sampled_utility.size() != K) {
        throw string("invalid utility size in SamplePermutationSingle.");
    }
    if (sampled_actions.size() != K) {
        throw string("invalid sampled_actions size in SamplePermutationSingle.");
    }
    vector<double> exp_values(K, 0.0);
    vector<bool> used(K, false);
    double sum_exp = 0.0;
    for (int k_1 = 0; k_1 < K; k_1++) {
        exp_values[k_1] = exp(sampled_utility[k_1] / tau);
        sum_exp += exp_values[k_1];
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double prob;
    for (int k_1 = 0; k_1 < K; k_1++) {
        prob = dist(gen) * sum_exp;
        bool flg = false;
        for (int k_2 = 0; k_2 < K; k_2++)
            if (!used[k_2]) {
                prob -= exp_values[k_2];
                if (prob < 1e-8) {
                    flg = true;
                    permutation.push_back(sampled_actions[k_2]);
                    sum_exp -= exp_values[k_2];
                    exp_values[k_2] = 0.0;
                    used[k_2] = true;
                    break;
                }
            }
        if (!flg) {
            for (int k_2 = K - 1; k_2 >= 0; k_2--)
                if (!used[k_2]) {
                    permutation.push_back(sampled_actions[k_2]);
                    sum_exp -= exp_values[k_2];
                    exp_values[k_2] = 0.0;
                    used[k_2] = true;
                }
        }
    }
    for (int k_1 = 0; k_1 < K; k_1++) {
        if (exp_values[k_1] > 1e-6) {
            throw string("error in SamplePermutationSingle.");
        }
    }
    return permutation;
}

double distance(vector<double> &a, vector<double> &b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_ENV_H
