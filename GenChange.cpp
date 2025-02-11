#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>

#include "FRW.h"

using namespace std;

int main() {
    ifstream file("paras/change_para.txt");
    if (!file) {
        cerr << "Can't open change_para" << std::endl;
        return 1;
    }
    vector<unsigned long long> T_list;
    vector<double> b_list;
    string line;
    if (getline(file, line)) {
        istringstream iss(line);
        unsigned long long T;
        while (iss >> T) {
            T_list.push_back(T);
        }
    }
    if (getline(file, line)) {
        istringstream iss(line);
        double b;
        while (iss >> b) {
            b_list.push_back(b);
        }
        b_list.push_back(1.0);
    }

    for (unsigned long long T: T_list) {
        for (double b: b_list) {
            for (int i = 0; i < 10; i++) {
                mt19937 gen(i);
                double sum = 0.0;
                string file_name = "possible_changes/" + to_string(T) + "_" + to_string(b) + "_" + to_string(i) + ".txt";
                std::uniform_real_distribution<> distrib(0.0, 1.0);
                double *all_change = new double[T];
                for (unsigned long long t = 0; t < T; t++) {
                    all_change[t] = distrib(gen);
                    sum += all_change[t];
                }
                double rate = pow(T, b) / sum;
                for (unsigned long long t = 0; t < T; t++) {
                    all_change[t] *= rate;
                }
                fwrite_double(file_name, all_change, T);
                delete[] all_change;
            }
        }
    }

}
