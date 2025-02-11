#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <queue>
#include "Parameters.h"
#include "Strategy.h"
#include "ChernoffEst.h"
#include "Env.h"

using namespace std;

class Tensor {
public:
    int dim;
    std::vector<int> shape;
    std::vector<double> data;

    Tensor(const std::vector<double> data_, const std::vector<int> &shape_) : shape(shape_), data(data_) {
        dim = shape_.size();
    }

    ~Tensor() = default;

    Tensor() : dim{0} {}

    double operator[](const std::vector<int> &idx) {
        int index = 0;
        for (int i = 0; i < dim; ++i) {
            index = index * shape[i] + idx[i];
        }
        return data[index];
    }
};


Tensor SetMatrix(mt19937 &gen) {
    vector<double> data;
    uniform_real_distribution<double> dist(-utility_abs_bound, utility_abs_bound);
    for (int i = 0; i < pow(action_size, player_num); ++i) {
        data.push_back(dist(gen));
    }
    vector<int> shape(player_num, action_size);
    return Tensor(data, shape);
}


class Player {
public:
    vector<double> strategy;
    vector<double> alg_strategy;
    vector<double> accumulated_utility;
    vector<double> emp_utility;
    vector<double> accumulated_emp_utility;
    double bandit_reward;
    double fullinfo_reward;
    vector<int> m1_cnt;
    queue<vector<double>> estimation_exp_list;
    vector<double> estimation_exp_sum;
    vector<vector<int>> permutations;
    vector<double> est_avg_utility;
    vector<vector<double>> estimated_emp_utility_list;
    vector<vector<int>> n_list;
    vector<int> sum_n;
    vector<double> estimated_emp_utility_cum, previous_cum_estimate;
    vector<int> emp_cum_num;

    Player(int action_size, mt19937 &gen) {
        strategy = GetInitStrategy(1.0, gamma_v, vector<double>(action_size, 1.0 / action_size), gen);
        alg_strategy = GetInitStrategy(1.0, gamma_v, vector<double>(action_size, 1.0 / action_size), gen);
        accumulated_utility = vector<double>(action_size, 0.0);
        vector<vector<int>> permutations;
        m1_cnt = vector<int>(action_size - 1, 0);
        estimation_exp_sum = vector<double>(action_size - 1, 0.0);
        bandit_reward = 0.0;
        fullinfo_reward = 0.0;
        for (int i = 0; i < action_size; i++) {
            est_avg_utility.push_back(0.0);
            sum_n.push_back(0);
        }
        estimated_emp_utility_list.push_back(vector<double>(action_size, 0.0));
        n_list.push_back(vector<int>(action_size, 0));
        emp_utility = vector<double>(action_size, 0.0);
        accumulated_emp_utility = vector<double>(action_size, 0.0);
        estimated_emp_utility_cum = vector<double>(action_size, 0.0);
        previous_cum_estimate = vector<double>(action_size, 0.0);
        emp_cum_num = vector<int>(action_size, 0);
    }

    ~Player() = default;

};

int action_set_size_cum[15];
bool is_action_set_init = false;

void init_utility_calc() {
    action_set_size_cum[0] = 1;
    for (int i = 1; i <= player_num; i++) {
        action_set_size_cum[i] = action_set_size_cum[i - 1] * action_size;
    }
}

void CalculateSingleUtility(int player, vector<Player> &players, Tensor &matrix, vector<double> &single_utility) {
    if (!is_action_set_init) {
        init_utility_calc();
        is_action_set_init = true;
    }
    for (int action = 0; action < action_size; ++action) {
        double utility = 0, prob = 0;
        for (int joint_action = 0, S = action_set_size_cum[player_num - 1], div; joint_action < S; joint_action++) {
            div = action_set_size_cum[player_num - player - 1];
            int all_joint_action = ((joint_action / div) * action_size + action) * div + joint_action % div;
            prob = 1.0;
            for (int i = player_num - 1, x = all_joint_action; i >= 0; i--, x /= action_size)
                if (i != player) {
                    prob *= players[i].strategy[x % action_size];
                }
            utility += matrix.data[all_joint_action] * prob;
        }
        single_utility[action] = utility;
    }
}

double CalculateSingleUtilityBandit(vector<int> joint_action, Tensor &matrix) {
    if (!is_action_set_init) {
        init_utility_calc();
        is_action_set_init = true;
    }
    return matrix[joint_action];
}

int main() {
    string para_file_name = SetParameters();
    mt19937 gen(exp_idx);
    try {
        if (is_full_info) {
            string result_file_fullinfo =
                    "results/CCE_Average/regrets_" + para_file_name + "/" + "fullinfo_" + to_string(exp_idx) + ".txt";
            std::ofstream results_fullinfo(result_file_fullinfo);
            if (!results_fullinfo.is_open()) {
                throw string("Failed to open the result file!");
            }
            vector<Tensor> matrices;
            for (int player; player < player_num; player++) {
                matrices.push_back(SetMatrix(gen));
            }
            vector<Player> players;
            for (int i = 0; i < player_num; ++i) {
                players.push_back(Player(action_size, gen));
            }
            vector<double> fullinfo_regrets;
            vector<int> timestep;
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<vector<double>> single_utilities(player_num, vector<double>(action_size, 0.0));
            vector<vector<int>> sampled_actions(player_num, vector<int>(action_size, -1));
            for (int player = 0; player < player_num; player++) {
                for (int action = 0; action < action_size; action++) {
                    sampled_actions[player][action] = action;
                }
            }
            int print_per_round = T / print_num;
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }
                for (int player = 0; player < player_num; player++) {
                    CalculateSingleUtility(player, players, matrices[player], single_utilities[player]);
                    vector<double> single_average_utility;
                    for (int i = 0; i < action_size; i++) {
                        players[player].accumulated_utility[i] += single_utilities[player][i];
                        single_average_utility.push_back(players[player].accumulated_utility[i] / (t + 1));
                    }
                    auto new_permutation = SamplePermutationSingle(single_average_utility, sampled_actions[player], gen);
                    players[player].permutations.push_back(new_permutation);
                    update_single_time_estimation(new_permutation, players[player].estimation_exp_list,
                                                  players[player].estimation_exp_sum, players[player].m1_cnt);
                    for (int i = 0; i < action_size; i++) {
                        players[player].fullinfo_reward += single_utilities[player][i] * players[player].strategy[i];
                    }
                }
                for (int player = 0; player < player_num; player++) {
                    vector<double> esimated_utility = chernoff_estimate(t, players[player].estimation_exp_sum,
                                                                        players[player].m1_cnt);
                    UpdateStrategyFTRL(players[player].strategy, esimated_utility, 1.0, gamma_v,
                                       lowerbound, t);
                }
                if (t % print_per_round == 0) {
                    double avg_fullinfo_regret = 0.0;
                    for (int player = 0; player < player_num; player++) {
                        double max_best_reward = players[player].accumulated_utility[0];
                        for (int i = 1; i < action_size; i++) {
                            if (max_best_reward < players[player].accumulated_utility[i]) {
                                max_best_reward = players[player].accumulated_utility[i];
                            }
                        }
                        avg_fullinfo_regret += max(max_best_reward - players[player].fullinfo_reward, 0.0);
                    }
                    avg_fullinfo_regret /= player_num;
                    timestep.push_back(t);
                    fullinfo_regrets.push_back(avg_fullinfo_regret);
                }
            }
            write_results(timestep, fullinfo_regrets, results_fullinfo);
            results_fullinfo.close();
        } else {
            eta = 1e-6;
            string result_file_bandit =
                    "results/CCE_Average/regrets_" + para_file_name + "/" + "bandit_" + to_string(exp_idx) + ".txt";
            std::ofstream results_bandit(result_file_bandit);
            if (!results_bandit.is_open()) {
                throw string("Failed to open the result file!");
            }
            vector<Tensor> matrices;
            for (int player; player < player_num; player++) {
                matrices.push_back(SetMatrix(gen));
            }
            vector<Player> players;
            for (int i = 0; i < player_num; ++i) {
                players.push_back(Player(action_size, gen));
            }
            vector<double> bandit_regrets;
            vector<int> timestep;
            vector<double> lowerbound(action_size, 1.0 / action_size);
            vector<vector<double>> single_utilities_fullinfo(player_num, vector<double>(action_size, 0.0));
            vector<vector<double>> sampled_utilities(player_num, vector<double>(K, 0.0));
            vector<vector<int>> sampled_actions(player_num, vector<int>(K, -1));
            int print_per_round = T / print_num;
            for (int t = 1; t <= T; t++) {
                if (t % 1000 == 0) {
                    showProgressBar(t, T);
                }

                for (int player = 0; player < player_num; player++) {
                    sampled_actions[player] = SampleActions(players[player].strategy, gen);
                }

                for (int player = 0; player < player_num; player++) {
                    CalculateSingleUtility(player, players, matrices[player], single_utilities_fullinfo[player]);
                    for (int i = 0; i < action_size; i++) {
                        players[player].fullinfo_reward += single_utilities_fullinfo[player][i] * players[player].strategy[i];
                    }
                    for (int i = 0; i < action_size; i++) {
                        players[player].accumulated_utility[i] += single_utilities_fullinfo[player][i];
                    }
                    for (int i = 0; i < K; i++) {
                        players[player].sum_n[sampled_actions[player][i]]++;
                    }
                    for (int k = 0; k < K; k++) {
                        vector<int> idx;
                        for (int player_ = 0; player_ < player_num; player_++) {
                            idx.push_back(sampled_actions[player_][k]);
                        }
                        sampled_utilities[player][k] = CalculateSingleUtilityBandit(idx, matrices[player]);
                    }
                    auto new_permutation = SamplePermutationSingleBandit(sampled_utilities[player], sampled_actions[player], gen);
                    players[player].permutations.push_back(new_permutation);
                    update_single_time_estimation(new_permutation, players[player].estimation_exp_list,
                                                  players[player].estimation_exp_sum, players[player].m1_cnt);
                }
                for (int player = 0; player < player_num; player++) {
                    auto estimate = chernoff_estimate(t, players[player].estimation_exp_sum, players[player].m1_cnt);
                    bool flg = (t % M == 0);
                    for (int action = 0, S = players[player].n_list.size() - 1; action < action_size - 1; action++) {
                        if (players[player].n_list[S][action] + 10000 >= players[player].sum_n[action] && !flg) {
                            continue;
                        }
                        players[player].emp_cum_num[action]++;
                        players[player].estimated_emp_utility_cum[action] +=
                                (estimate[action] * players[player].sum_n[action] - players[player].estimated_emp_utility_list[S][action] * players[player].n_list[S][action]) /
                                (players[player].sum_n[action] - players[player].n_list[S][action]);

                    }
                    for (int action = 0; action < action_size - 1; action++)
                        if (players[player].emp_cum_num[action] > 0) {
                            players[player].est_avg_utility[action] =
                                    (players[player].previous_cum_estimate[action] + players[player].estimated_emp_utility_cum[action] / (players[player].emp_cum_num[action])) /
                                    double(players[player].estimated_emp_utility_list.size());
                        }
                    players[player].est_avg_utility[action_size - 1] = 0.0;
                    if (t % M == 0) {
                        players[player].estimated_emp_utility_list.push_back(estimate);
                        players[player].n_list.push_back(players[player].sum_n);
                        for (int action = 0; action < action_size - 1; action++) {
                            players[player].previous_cum_estimate[action] += players[player].estimated_emp_utility_cum[action] / players[player].emp_cum_num[action];
                            players[player].estimated_emp_utility_cum[action] = 0.0;
                            players[player].emp_cum_num[action] = 0;
                        }
                    }
                    players[player].strategy = UpdateStrategyFTRL2(players[player].alg_strategy, players[player].est_avg_utility, 1.0, gamma_v, lowerbound, t);
                }
                if (t % print_per_round == 0) {
                    double avg_fullinfo_regret = 0.0;
                    for (int player = 0; player < player_num; player++) {
                        double max_best_reward = players[player].accumulated_utility[0];
                        for (int i = 1; i < action_size; i++) {
                            if (max_best_reward < players[player].accumulated_utility[i]) {
                                max_best_reward = players[player].accumulated_utility[i];
                            }
                        }
                        avg_fullinfo_regret += max(max_best_reward - players[player].fullinfo_reward, 0.0);
                    }
                    avg_fullinfo_regret /= player_num;
                    timestep.push_back(t);
                    bandit_regrets.push_back(avg_fullinfo_regret);
                }
            }
            write_results(timestep, bandit_regrets, results_bandit);
            results_bandit.close();
        }
    } catch (const string &error_msg) {
        cerr << "Error: " << error_msg << endl;
    }
    return 0;
}