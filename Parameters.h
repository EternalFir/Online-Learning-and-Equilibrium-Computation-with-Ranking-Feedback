#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CONSTANTS_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CONSTANTS_H

#include <cmath>
#include <iostream>
#include <iomanip>
const double utility_abs_bound = 1.0;
const int print_num = 1000;

using namespace std;

void showProgressBar(unsigned long long progress, unsigned long long total) {
    const int barWidth = 50;
    float percent = static_cast<float>(progress) / total;
    int pos = static_cast<int>(barWidth * percent);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            std::cout << "=";
        } else if (i == pos) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << int(percent * 100.0) << "%\r";
    std::cout.flush();
}

extern double b;
extern unsigned long long T;
extern int action_size;
extern int K;
extern int m;
extern int M;
extern double tau;
extern double gamma_v;
extern double Delta;
extern double eta;
extern int exp_idx;
extern int player_num;
extern bool is_full_info;

string SetParameters() {
    cin >> action_size >> T >> K >> tau >> b >> m >> gamma_v >> player_num >> M >> is_full_info >> exp_idx;
    stringstream ss_tau, ss_b, ss_gamma;
    ss_tau << tau;
    ss_b << b;
    ss_gamma << gamma_v;
    string tau_str = ss_tau.str();
    string b_str = ss_b.str();
    string m_str = to_string(m);
    string M_str = to_string(M);
    string gamma_str = ss_gamma.str();
    string is_full_info_str = is_full_info ? "True" : "False";
    if (b_str == "1")
        b_str = "1.0";
    string para_file_name = to_string(action_size) + "_" + to_string(T) + '_' + to_string(K) + "_" +
                            tau_str + "_" + b_str + "_" + m_str + "_" + gamma_str + "_" + to_string(player_num) + "_" + M_str + "_" +
                            is_full_info_str;
    Delta = std::pow(T, b);
    eta = 1 / std::sqrt(T);

    return para_file_name;
}

void write_results(vector<int> &timestep, vector<double> &regrets,
                   ofstream &file_stream) {
    for (int i = 0; i < timestep.size(); i++) {
        file_stream << timestep[i] << " " << std::fixed << std::setprecision(6) << regrets[i] << '\n';
    }
}

#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_CONSTANTS_H
