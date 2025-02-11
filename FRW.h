#ifndef ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_FRW_H
#define ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_FRW_H

#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>

using namespace std;

double *fread_double(string file_name, unsigned long long T) {
    int fd = open(file_name.c_str(), O_RDONLY);
    if (fd == -1) {
        cout << "fread open file failed" << endl;
        exit(1);
    }
    double *data = (double *) mmap(NULL, T * sizeof(double), PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        cout << "mmap failed" << endl;
        exit(1);
    }
    close(fd);
    return data;
}

void fwrite_double(string file_name, double *data_in, unsigned long long T) {
    int fd = open(file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
        cout << "fwrite open file failed" << endl;
        exit(1);
    }
    if (ftruncate(fd, T * sizeof(double)) == -1) {
        perror("ftruncate failed");
        close(fd);
        return;
    }
    double *data = (double *) mmap(NULL, T * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        cout << "mmap failed" << endl;
        close(fd);
        exit(1);
    }
    for (unsigned long long t = 0; t < T; t++) {
        data[t] = data_in[t];
    }
    munmap(data, T * sizeof(double));
    close(fd);
    return;
}


#endif //ONLINE_LEARNING_AND_EQILIBRIUM_COMPUTATION_WITH_RANKING_FEEDBACK_FRW_H
