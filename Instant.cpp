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
    vector<double> accu_reward(action_size, 0.0);
    cout << para_file_name << endl;


    try {
        if (is_full_info) {
            double fullinfo_reward = 0.0;
            string result_file_fullinfo =
                    "results/Instant/regrets_" + para_file_name + "/" + "fullinfo_" + to_string(exp_idx) + ".txt";
            std::ofstream results_fullinfo(result_file_fullinfo);
            if (!results_fullinfo.is_open()) {
                throw string("Failed to open the result file!");
            }
            auto start = std::chrono::high_resolution_clock::now();
            vector<double> utility = GetInitUtility(action_size, 1.0, gen);
            vector<pair<double, double>> utility_range;
            for (int i = 0; i < action_size - 1; i++) {
                if (utility[i] >= 0.0) {
                    utility_range.push_back(pair<double, double>(2 * utility[i] - 1.0, 1.0));
                } else {
                    utility_range.push_back(pair<double, double>(-1.0, 2 * utility[i] + 1.0));
                }
            }
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<double> strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> fullinfo_regrets;
            vector<int> timestep;
            vector<int> m1_cnt(action_size - 1, 0);
            queue<vector<double>> estimation_exp_list;
            vector<double> estimation_exp_sum(action_size - 1, 0.0);
            int print_per_round = T / print_num;
            vector<int> action_set;
            for (int i = 0; i < action_size; i++) {
                action_set.push_back(i);
            }
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }
                if (b != 0) {
                    UpdateUtility(utility, utility_range, all_change[t], gen);
                }
                vector<int> sampled_actions = action_set;
                vector<int> new_permutation = SamplePermutationSingle(utility, sampled_actions, gen);
                permutations.push_back(new_permutation);
                update_single_time_estimation(new_permutation, estimation_exp_list, estimation_exp_sum, m1_cnt);

                for (int i = 0; i < action_size; i++) {
                    accu_reward[i] += utility[i];
                }
                for (int i = 0; i < action_size; i++) {
                    fullinfo_reward += utility[i] * strategy[i];
                }
                vector<double> estimated_utility = chernoff_estimate(t, estimation_exp_sum, m1_cnt);
                UpdateStrategyPGD(strategy, estimated_utility, 1.0, gamma_v, lowerbound);
                if (t % print_per_round == 0) {
                    double max_best_reward = accu_reward[0];
                    for (int i = 1; i < action_size; i++) {
                        if (max_best_reward < accu_reward[i]) {
                            max_best_reward = accu_reward[i];
                        }
                    }
                    timestep.push_back(t);
                    fullinfo_regrets.push_back(max_best_reward - fullinfo_reward);
                }
            }
            write_results(timestep, fullinfo_regrets, results_fullinfo);
            results_fullinfo.close();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            cout << "time: " << duration.count() << endl;
        } else {
            double bandit_reward = 0.0;
            string result_file_bandit =
                    "results/Instant/regrets_" + para_file_name + "/" + "bandit_" + to_string(exp_idx) + ".txt";
            std::ofstream results_bandit(result_file_bandit);
            if (!results_bandit.is_open()) {
                throw string("Failed to open the result file!");
            }
            auto start = std::chrono::high_resolution_clock::now();
            vector<double> utility = GetInitUtility(action_size, 1.0, gen);
            vector<pair<double, double>> utility_range;
            for (int i = 0; i < action_size - 1; i++) {
                if (utility[i] >= 0.0) {
                    utility_range.push_back(pair<double, double>(2 * utility[i] - 1.0, 1.0));
                } else {
                    utility_range.push_back(pair<double, double>(-1.0, 2 * utility[i] + 1.0));
                }
            }
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<double> strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> alg_strategy = GetInitStrategy(1.0, gamma_v, lowerbound, gen);
            vector<double> bandit_regrets;
            vector<int> timestep;
            vector<int> m1_cnt(action_size - 1, 0);
            queue<vector<double>> estimation_exp_list;
            vector<double> estimation_exp_sum(action_size - 1, 0.0);
            int print_per_round = T / print_num;
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }
                if (b != 0) {
                    UpdateUtility(utility, utility_range, all_change[t], gen);
                }
                vector<int> sampled_actions = SampleActions(strategy, gen);
                vector<int> new_permutation = SamplePermutationSingle(utility, sampled_actions, gen);
                permutations.push_back(new_permutation);
                update_single_time_estimation(new_permutation, estimation_exp_list, estimation_exp_sum, m1_cnt);

                for (int i = 0; i < action_size; i++) {
                    accu_reward[i] += utility[i];
                }
                for (int i = 0; i < K; i++) {
                    bandit_reward += utility[sampled_actions[i]];
                }
                vector<double> estimated_utility = chernoff_estimate(t, estimation_exp_sum, m1_cnt);
                strategy = UpdateStrategyPGD2(alg_strategy, estimated_utility, 1.0, gamma_v, lowerbound, t);
                if (t % print_per_round == 0) {
                    double max_best_reward = accu_reward[0];
                    for (int i = 1; i < action_size; i++) {
                        if (max_best_reward < accu_reward[i]) {
                            max_best_reward = accu_reward[i];
                        }
                    }
                    timestep.push_back(t);
                    bandit_regrets.push_back(max_best_reward - (bandit_reward / K));
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



