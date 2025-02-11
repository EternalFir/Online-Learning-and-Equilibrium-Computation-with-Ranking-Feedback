import multiprocessing
import sys
import os
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

import threading
import subprocess


class Parameter():
    def __init__(self, partial_run={}):
        self.param = {'algo': ['Instant', 'Average', "CCE_Instant", 'CCE_Average'],
                      'is_full_info': [False, True],
                      'player_num': [2],
                      'action_size': [10],
                      'T': [10000000],
                      'K': [3, 5, 10],
                      'tau': [0.5, 1, 2],
                      'm': [50000, 100000, 150000],
                      'gamma': [0.1, 0.05, 0.01],
                      'M': [3000000, 5000000, 10000000],
                      'b': [0.3, 0.5, 0.7],
                      'exp_idx': [_ for _ in range(10)]
                      }

        self.mask_list = [
            [['algo', ['CCE_Instant', 'CCE_Average']], 'b', [1.0]],
            [['algo', ['Average'], 'is_full_info', [True]], 'b', [1.0]],
            [['algo', ['Average'], 'is_full_info', [False]], 'b', [0.1, 0.2, 0.3]],
            [['is_full_info', [True]], 'K', [10]],
            [['is_full_info', [True]], 'M', [10000000]],
            [['algo', ['Instant', 'CCE_Instant']], 'M', [10000000]],
            [['algo', ['Instant', 'Average']], 'player_num', [1]],
        ]
        # type_A, key_list_A, type_B, key_list_B.  Once key_list_A is selected, key_list_B is the only choice
        self.param_list = []
        self.Set_Param(0, {})

        self.n = len(self.param_list)

        self.active_idx = []
        for i in range(self.n):
            flag = True
            for k in partial_run:
                if self.param_list[i][k] not in partial_run[k]:
                    flag = False
                    break
            if flag:
                self.active_idx.append(i)
        self.n = len(self.active_idx)
        print(self.n)

    def Set_Param(self, k, param):
        if k == len(list(self.param.keys())):
            self.param_list.append(param.copy())
            return

        name = list(self.param.keys())[k]
        param_list = self.param[name]

        if name in param.keys():
            self.Set_Param(k + 1, param)
            return

        for mask in self.mask_list:
            flg = True
            for i in range(0, len(mask[0]), 2):
                if mask[0][i] in param.keys() and param[mask[0][i]] not in mask[0][i + 1]:
                    flg = False
                    break
            if flg and name == mask[1]:
                param_list = mask[2]
            if flg and mask[1] in param.keys() and param[mask[1]] not in mask[2]:
                return
        for v in param_list:
            param[name] = v
            self.Set_Param(k + 1, param)
        del param[name]

    def get_param(self, idx):
        if type(idx) == int:
            return self.param_list[self.active_idx[idx]]
        elif type(idx) == dict:
            param_idx_list = []
            for i in range(self.n):
                flag = True
                for k, v in idx.items():
                    if self.param_list[i][k] != v:
                        flag = False
                        break
                if flag:
                    param_idx_list.append(i)
            return np.array(param_idx_list)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='Instant')
    return parser.parse_args()

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    print(current_path)

    task_idx = int(sys.argv[1])
    node_num = int(sys.argv[2])
    param = Parameter()

    os.makedirs(f'{current_path}/paras', exist_ok=True)
    if node_num == 0:
        possible_T = param.param['T']
        possible_b = param.param['b']
        possible_b.append(0.1)
        possible_b.append(0.2)
    
        with open(f'{current_path}/paras/change_para.txt', 'w') as f:
            write = ""
            for t in possible_T:
                write += str(t) + " "
            print(write, file=f)
            write = ""
            for b in possible_b:
                write += str(b) + " "
            print(write, file=f)
        os.makedirs(f"{current_path}/possible_changes", exist_ok=True)
        os.system(f"./GenChange")
        exit(0)

    for i in range(task_idx, param.n, node_num):
        w = param.get_param(i)
        os.makedirs(f'{current_path}/paras', exist_ok=True)
        os.makedirs(f"{current_path}/results/{w['algo']}", exist_ok=True)

        name_file = ''
        for k in [w['action_size'], w['T'], w['K'], w['tau'], w['b'], w['m'], w['gamma'], w['player_num'], w['M'],
                  w['is_full_info']]:
            name_file += str(k) + '_'
        name_file = name_file[:-1]

        os.makedirs(f"{current_path}/results/{w['algo']}/regrets_{name_file}", exist_ok=True)
        print(w)
        with open(f'{current_path}/paras/{i}.in', 'w') as f:
            print(w['action_size'], w['T'], w['K'], w['tau'], w['b'], w['m'], w['gamma'], w['player_num'], w['M'],
                  int(w['is_full_info']), w['exp_idx'], file=f)
        os.system(str(f"{current_path}/{w['algo']} < {current_path}/paras/{i}.in"))
