#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <queue>

#include <chrono>

#include "Strategy.h"
#include "ChernoffEst.h"
#include "Parameters.h"
#include "Env.h"
#include "FRW.h"

using namespace std;

int main() {
    string para_file_name = SetParameters();
    mt19937 gen(exp_idx);
    string change_file_name = "possible_changes/" + to_string(T) + "_" + to_string(b) + "_" + to_string(exp_idx) + ".txt";
    double *all_change = fread_double(change_file_name, T);
    vector<vector<int>> permutations;
    double bandit_reward = 0.0;
    double fullinfo_reward = 0.0;
    try {
        if (is_full_info) {
            string result_file_fullinfo =
                    "results/Average/regrets_" + para_file_name + "/" + "fullinfo_" + to_string(exp_idx) + ".txt";
            std::ofstream results_fullinfo(result_file_fullinfo);
            if (!results_fullinfo.is_open()) {
                throw string("Failed to open the result file!");
            }
            auto start = std::chrono::high_resolution_clock::now();
            vector<double> utility = GetInitUtility(action_size, 1.0, gen);
            vector<pair<double, double>>
                    utility_range;
            for (int i = 0; i < action_size - 1; i++) {
                if (utility[i] >= 0.0) {
                    utility_range.push_back(pair<double, double>(2 * utility[i] - 1.0, 1.0));
                } else {
                    utility_range.push_back(pair<double, double>(-1.0, 2 * utility[i] + 1.0));
                }
            }
            vector<double> average_utility(action_size, 0.0);
            vector<double> accumulated_utility(action_size, 0.0);
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<double> strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> fullinfo_regrets;
            vector<int> timestep;
            vector<int> action_set;
            for (int i = 0; i < action_size; i++) {
                action_set.push_back(i);
            }
            vector<int> m1_cnt(action_size - 1, 0);
            queue<vector<double>> estimation_exp_list;
            vector<double> estimation_exp_sum(action_size - 1, 0.0);
            int print_per_round = T / print_num;
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }
                UpdateUtility(utility, utility_range, all_change[t], gen);
                for (int i = 0; i < action_size; i++) {
                    accumulated_utility[i] += utility[i];
                    average_utility[i] = accumulated_utility[i] / (t + 1);
                }
                vector<int> sampled_actions = action_set;
                vector<int> new_permutation = SamplePermutationSingle(average_utility, sampled_actions, gen);
                permutations.push_back(new_permutation);
                update_single_time_estimation(new_permutation, estimation_exp_list, estimation_exp_sum, m1_cnt);
                for (int i = 0; i < action_size; i++) {
                    fullinfo_reward += utility[i] * strategy[i];
                }
                vector<double> estimated_avg_utility = chernoff_estimate(t, estimation_exp_sum, m1_cnt);
                UpdateStrategyFTRL(strategy, estimated_avg_utility, 1.0, gamma_v, lowerbound, t);
                if (t % print_per_round == 0) {
                    double max_best_reward = accumulated_utility[0];
                    for (int i = 1; i < action_size; i++) {
                        if (max_best_reward < accumulated_utility[i]) {
                            max_best_reward = accumulated_utility[i];
                        }
                    }
                    fullinfo_regrets.push_back(max_best_reward - fullinfo_reward);
                    timestep.push_back(t);
                }
            }
            write_results(timestep, fullinfo_regrets, results_fullinfo);
            results_fullinfo.close();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            cout << "time: " << duration.count() << endl;

        } else {
            eta = 1e-6;
            string result_file_bandit =
                    "results/Average/regrets_" + para_file_name + "/" + "bandit_" + to_string(exp_idx) + ".txt";
            std::ofstream results_bandit(result_file_bandit);
            if (!results_bandit.is_open()) {
                throw string("Failed to open the result file!");
            }
            auto start = std::chrono::high_resolution_clock::now();
            vector<double> utility = GetInitUtility(action_size, 1.0, gen);
            vector<pair<double, double>>
                    utility_range;
            for (int i = 0; i < action_size - 1; i++) {
                if (utility[i] >= 0.0) {
                    utility_range.push_back(pair<double, double>(2 * utility[i] - 1.0, 1.0));
                } else {
                    utility_range.push_back(pair<double, double>(-1.0, 2 * utility[i] + 1.0));
                }
            }
            vector<double> emp_utility(action_size, 0.0);
            vector<double> accumulated_emp_utility(action_size, 0.0);
            vector<double> accumulated_utility(action_size, 0.0);
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<double> strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> alg_strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> bandit_regrets;
            vector<int> timestep;
            vector<int> m1_cnt(action_size - 1, 0);
            queue<vector<double>> estimation_exp_list;
            vector<double> estimation_exp_sum(action_size - 1, 0.0);
            vector<double> est_avg_utility(action_size, 0.0);
            vector<vector<double>> estimated_emp_utility_list;
            vector<vector<int>> n_list;
            vector<int> sum_n(action_size, 0);
            vector<int> emp_cum_num(action_size, 0);
            vector<double> estimated_emp_utility_cum(action_size, 0.0);
            vector<double> previous_cum_estimate(action_size, 0.0);
            int print_per_round = T / print_num;
            estimated_emp_utility_list.push_back(vector<double>(action_size, 0.0));
            n_list.push_back(vector<int>(action_size, 0));
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }
                UpdateUtility(utility, utility_range, all_change[t], gen);
                for (int i = 0; i < action_size; i++) {
                    accumulated_utility[i] += utility[i];
                }
                vector<int> sampled_actions = SampleActions(strategy, gen);
                for (int i = 0; i < K; i++) {
                    sum_n[sampled_actions[i]]++;
                    accumulated_emp_utility[sampled_actions[i]] += utility[sampled_actions[i]];
                }
                for (int i = 0; i < action_size - 1; i++) {
                    emp_utility[i] = accumulated_emp_utility[i] / sum_n[i];
                }
                vector<int> new_permutation = SamplePermutationSingle(emp_utility, sampled_actions, gen);
                permutations.push_back(new_permutation);
                update_single_time_estimation(new_permutation, estimation_exp_list, estimation_exp_sum, m1_cnt);
                for (int i = 0; i < K; i++) {
                    bandit_reward += utility[sampled_actions[i]];
                }
                auto estimate = chernoff_estimate(t, estimation_exp_sum, m1_cnt);
                bool flg = (t % M == 0);
                for (int action = 0, S = n_list.size() - 1; action < action_size - 1; action++) {
                    if (n_list[S][action] + 10000 >= sum_n[action] && !flg) {
                        continue;
                    }
                    emp_cum_num[action]++;
                    estimated_emp_utility_cum[action] +=
                            (estimate[action] * sum_n[action] - estimated_emp_utility_list[S][action] * n_list[S][action]) / (sum_n[action] - n_list[S][action]);
                }
                for (int action = 0; action < action_size - 1; action++)
                    if (emp_cum_num[action] > 0) {
                        est_avg_utility[action] =
                                (previous_cum_estimate[action] + estimated_emp_utility_cum[action] / (emp_cum_num[action])) / double(estimated_emp_utility_list.size());
                    }
                est_avg_utility[action_size - 1] = 0.0;
                if (t % M == 0) {
                    auto avg_utility = accumulated_utility;
                    for (int i = action_size - 1; i >= 0; i--) {
                        avg_utility[i] /= t;
                        avg_utility[i] -= avg_utility[action_size - 1];
                    }
                    cout << t << " " << distance(avg_utility, est_avg_utility) << endl;
                    estimated_emp_utility_list.push_back(estimate);
                    n_list.push_back(sum_n);
                    for (int action = 0; action < action_size - 1; action++) {
                        previous_cum_estimate[action] += estimated_emp_utility_cum[action]
                                                         / emp_cum_num[action];
                        estimated_emp_utility_cum[action] = 0.0;
                        emp_cum_num[action] = 0;
                    }
                }
                strategy = UpdateStrategyFTRL2(alg_strategy, est_avg_utility, 1.0, gamma_v, lowerbound, t);
                if (t % print_per_round == 0) {
                    double max_best_reward = accumulated_utility[0];
                    for (int i = 1; i < action_size; i++) {
                        if (max_best_reward < accumulated_utility[i]) {
                            max_best_reward = accumulated_utility[i];
                        }
                    }
                    bandit_regrets.push_back(max_best_reward - (bandit_reward / K));
                    timestep.push_back(t);
                }
            }
            write_results(timestep, bandit_regrets, results_bandit);
            results_bandit.close();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            cout << "time: " << duration.count() << endl;
        }

    } catch (const string &error_msg) {
        cerr << "Error: " << error_msg << endl;
    }


    return 0;
}



