#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_STRATEGY_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_STRATEGY_H

#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

#include "Parameters.h"
#include "Projection.h"

using namespace std;

vector<double>
GetInitStrategy(const double &abs_bound, const double &gamma, const vector<double> &lowerbound, mt19937 &gen) {
    vector<double> strategy(action_size, 1.0 / action_size);
    return strategy;
}

vector<int> SampleActions(vector<double> &strategy, mt19937 &gen) {
    vector<int> actions(K, 0);
    if (!is_full_info) {
        discrete_distribution<int> dist(strategy.begin(), strategy.end());
        for (int i = 0; i < K; i++) {
            actions[i] = dist(gen);
        }
    } else {
        vector<int> permutations(action_size, 0);
        for (int i = 0; i < action_size; i++) permutations[i] = i;
        shuffle(permutations.begin(), permutations.end(), gen);
        for (int i = 0; i < K; i++) {
            actions[i] = permutations[i];
        }
    }
    return actions;
}

void UpdateStrategyPGD(vector<double> &strategy, vector<double> &estimated_utility, const double &abs_bound,
                       const double &gamma,
                       const vector<double> &lowerbound) {
    for (int i = 0; i < action_size; i++) {
        strategy[i] += eta * estimated_utility[i];
    }
    SparseMax(strategy, gamma, lowerbound);
    return;
}

vector<double> UpdateStrategyPGD2(vector<double> &alg_strategy, vector<double> &estimated_utility, const double &abs_bound,
                                  const double &gamma, const vector<double> &lowerbound, unsigned long long t) {
    for (int i = 0; i < action_size; i++) {
        alg_strategy[i] += eta * estimated_utility[i];
    }
    SparseMax(alg_strategy, 0.0, lowerbound);
    vector<double> strategy_out;
    for (int i = 0; i < action_size; i++) {
        strategy_out.push_back((1 - gamma) * alg_strategy[i] + gamma * lowerbound[i]);
    }
    if (t % 100000 == 0) {
        for (int i = 0; i < action_size; i++) {
            if (strategy_out[i] < (gamma / action_size) - 1e-6) {
                throw (string("PGD update strategy error 1 at time ") + to_string(t));
            }
        }
        double sum = 0.0;
        for (int i = 0; i < action_size; i++) {
            sum += strategy_out[i];
        }
        if (abs(sum - 1.0) > 1e-6) {
            throw (string("PGD update strategy error 2 at time ") + to_string(t));
        }
    }
    return strategy_out;
}

void UpdateStrategyFTRL(vector<double> &strategy, vector<double> &estimated_avg_utility, const double &abs_bound,
                        const double &gamma,
                        const vector<double> &lowerbound, unsigned long long t) {
    for (int i = 0; i < action_size; i++) {
        strategy[i] = 2 * eta * estimated_avg_utility[i] * t;
    }
    SparseMax(strategy, gamma, lowerbound);
    return;
}

vector<double> UpdateStrategyFTRL2(vector<double> &alg_strategy, vector<double> &estimated_avg_utility, const double &abs_bound,
                                   const double &gamma, const vector<double> &lowerbound, unsigned long long t) {
    for (int i = 0; i < action_size; i++) {
        alg_strategy[i] = 2 * eta * estimated_avg_utility[i] * t;
    }
    SparseMax(alg_strategy, 0.0, lowerbound);
    vector<double> strategy_out;
    for (int i = 0; i < action_size; i++) {
        strategy_out.push_back((1 - gamma) * alg_strategy[i] + gamma * lowerbound[i]);
    }
    if (t % 100000 == 0) {
        for (int i = 0; i < action_size; i++) {
            if (strategy_out[i] < (gamma / action_size) - 1e-6) {
                throw (string("FTRL update strategy error 1 at time ") + to_string(t));
            }
        }
        double sum = 0.0;
        for (int i = 0; i < action_size; i++) {
            sum += strategy_out[i];
        }
        if (abs(sum - 1.0) > 1e-6) {
            throw (string("FTRL update strategy error 2 at time ") + to_string(t));
        }
    }
    return strategy_out;
}

#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_STRATEGY_H
