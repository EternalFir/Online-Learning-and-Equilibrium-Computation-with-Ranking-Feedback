#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_PROJECTION_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_PROJECTION_H

#include <vector>

using namespace std;

void DirectProj(vector<double> &utility, const double abs_bound) {
    for (int i = 0; i < utility.size(); i++) {
        if (utility[i] > abs_bound   or isnan(utility[i])) {
            utility[i] = abs_bound;
        } else if (utility[i] < -abs_bound){
            utility[i] = -abs_bound;
        } else;
    }
    return;
}

void DirectProj(vector<double> &value, const vector<pair<double,double>> &each_range) {
    for (int i=0;i<each_range.size();i++){
        if (value[i]> each_range[i].second or isnan(value[i])) {
            value[i] = each_range[i].second;
        } else if (value[i] < each_range[i].first){
            value[i] = each_range[i].first;
        } else;
    }
    return;
}

void UtilityZero(vector<double>& utility){
    for (int i = 0; i < utility.size(); i++) {
        utility[i]-= utility[utility.size()-1];
    }
    return;
}

void SparseMax(vector<double> &strategy, const double &gamma, const vector<double> &lowerbound) {
    unsigned int n = strategy.size();
    vector<double> aux(n);
    for (int i = 0; i < n; i++) aux[i] = strategy[i];
    sort(aux.begin(), aux.end());

    double tau_s = 0.0, C = -aux[0] + (1.0 - gamma) / double(n), cur_sum = 0.0;
    for (int i = 0; i < n; ++i)
        aux[i] += C,
                cur_sum += aux[i];
    for (int i = 0; i < n; ++i) {
        double sum = -(1.0 - gamma) + cur_sum;
        if (sum < aux[i] * (n - i)) {
            tau_s = sum / (n - i);
            break;
        }
        cur_sum -= aux[i];
    }
    cur_sum = 0.0;
    for (int i = 0; i < n; ++i)
        strategy[i] = std::max(strategy[i] - tau_s + C, 0.0) + gamma * lowerbound[i],
                cur_sum += strategy[i];
    for (int i = 0; i < n; ++i) strategy[i] /= cur_sum;
    return;
}

#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_PROJECTION_H
