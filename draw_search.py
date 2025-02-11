import multiprocessing
import sys
import os
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from distribute import Parameter
from multiprocessing import Pool

import matplotlib as mpl

mpl.use("pgf")

import threading
import subprocess
import copy

plt.rcParams.update({
    "axes.linewidth": 1.5,
    "font.family":
        "serif",  # use serif/main font for text elements
    "text.usetex":
        True,  # use inline math for ticks
    "pgf.rcfonts":
        False,  # don't setup fonts from rc parameters
    "pgf.preamble": """
        \\usepackage{times}
        \\usepackage[T1]{fontenc}
        \\usepackage{bm}
        \\usepackage{amsmath}
        \\usepackage{amssymb}
        \\usepackage{tikz}
        \\usepackage{inconsolata}
    """,
    "pgf.texsystem": "pdflatex",
})

def draw_regret(para, rounds, is_plot=False):
    regrets = []
    times = []
    if para["is_full_info"]:
        type = "fullinfo"
    else:
        type = "bandit"
    algo = para['algo']
    for idx in range(rounds):
        x = []
        y = []
        name_file = ''
        for k in [para['action_size'], para['T'], para['K'], para['tau'], para['b'], para['m'], para['gamma'],
                  para['player_num'], para['M'], para['is_full_info']]:
            name_file += str(k) + '_'
        name_file = name_file[:-1]

        file_path = str('results/' + algo + '/regrets_' + name_file + "/" + str(type) + "_" + str(idx) + '.txt')

        if not is_plot:
            with open(file_path, 'rb') as file:
                file.seek(0, 2)  # Move the cursor to the end of the file
                cursor = file.tell()
                last_char = ''
                flg = 0
                while cursor > 0:
                    cursor -= 1
                    file.seek(cursor)
                    last_char = file.read(1)
                    if last_char.isspace():  # Check if the character is a space
                        flg += 1
                    if flg == 2:
                        break
                last_float = file.read().strip()
                x.append(0)
                y.append(float(last_float))
        else:
            with open(file_path) as f:
                lines = f.readlines()
                if not lines:
                    print(f"File {file_path} is empty. Skipping.")
                    continue
                for line in lines:
                    line = line.split()
                    x.append(int(line[0]))
                    y.append(float(line[1]))
        regrets.append(y)
        times.append(x)

    times = np.array(times) + 1
    regrets = np.array(regrets)
    if para["algo"].startswith("CCE") and is_plot:
        regrets /= times
        regrets *= 2
    mean_regret = np.mean(regrets, axis=0)
    if not is_plot:
        return mean_regret[len(mean_regret) - 1]
    std_regret = np.std(regrets, axis=0)
    conf_interval = 1.96 * std_regret / np.sqrt(regrets.shape[0])
    os.makedirs("./figures/" + algo, exist_ok=True)
    plt.plot(times[0], mean_regret, color='blue')
    plt.fill_between(times[0], mean_regret - conf_interval, mean_regret + conf_interval, color='blue', alpha=0.3,
                     label='95% confidence interval')
    plt.title("Regret of " + algo)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.grid(True)
    plt.savefig('./figures/' + algo + '/regrets_' + name_file + "_" + str(type) + '.png')
    plt.close()
    return mean_regret[len(mean_regret) - 1]


def draw_multiple_regrets(para_to_draw, para_search_now, rounds, color_list, linestyle_list):
    plt.figure()
    for idx, para in enumerate(para_to_draw):
        regrets = []
        times = []
        if para["is_full_info"]:
            type = "fullinfo"
        else:
            type = "bandit"
        algo = para['algo']
        for round_idx in range(rounds):
            x = []
            y = []
            name_file = ''
            for k in [para['action_size'], para['T'], para['K'], para['tau'], para['b'], para['m'], para['gamma'],
                      para['player_num'], para["M"], para['is_full_info']]:
                name_file += str(k) + '_'
            name_file = name_file[:-1]
            file_path = str(
                'results/' + algo + '/regrets_' + name_file + "/" + str(type) + "_" + str(round_idx) + '.txt')
            with open(file_path) as f:
                lines = f.readlines()
                if not lines:
                    print(f"File {file_path} is empty. Skipping.")
                    continue
                for line in lines:
                    line = line.split()
                    x.append(int(line[0]))
                    y.append(float(line[1]))
            regrets.append(y)
            times.append(x)
        times = np.array(times) + 1
        regrets = np.array(regrets)
        if para["algo"].startswith("CCE"):
            regrets /= times * 2
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)
        conf_interval = 1.96 * std_regret / np.sqrt(regrets.shape[0])

        if para_search_now == "tau":
            plt.plot(times[0], mean_regret, color=color_list[idx], linestyle=linestyle_list[idx],
                     label=(r"$\tau$" + "=" + str(para[para_search_now])))
        else:
            plt.plot(times[0], mean_regret, color=color_list[idx], linestyle=linestyle_list[idx],
                     label=(para_search_now + "=" + str(para[para_search_now])))

        plt.fill_between(times[0], mean_regret - conf_interval, mean_regret + conf_interval, color=color_list[idx],
                         alpha=0.3)

    plt_name_para = {
        "algo": para_to_draw[0]['algo'],
        "K": para_to_draw[0]['K'],
        "tau": para_to_draw[0]['tau'],
        "b": para_to_draw[0]['b'],
        "is_full_info": para_to_draw[0]['is_full_info']
    }
    plot_title = ""
    if plt_name_para["algo"] == "Instant" or plt_name_para["algo"] == "CCE_Instant":
        plot_title += "InstUtil Rank "
    else:
        plot_title += "AvgUtil Rank "
    if plt_name_para["algo"].startswith("CCE"):
        plot_title += "Game "
    if plt_name_para['is_full_info']:
        plot_title += "with Full-information Feedback ("
    else:
        plot_title += "with Bandit Feedback ("
    store_name = plot_title
    for key in plt_name_para.keys():
        if key != para_search_now and key != "algo" and key != "is_full_info" and (
                not (key == 'b' and plt_name_para['algo'].startswith('CCE'))):
            if key == 'tau':
                plot_title = plot_title + r"$\tau$" + "=" + str(plt_name_para[key]) + ", "
            else:
                plot_title = plot_title + key + "=" + str(plt_name_para[key]) + ", "
            store_name = store_name + key + "_" + str(plt_name_para[key]) + " "
    plot_title = plot_title[:-2]
    plot_title += ")"
    fontsize = 15
    plt.title(plot_title, fontsize=fontsize)
    plt.title(plot_title, fontsize=fontsize)
    plt.xlabel("t", fontsize=fontsize)
    if para["algo"].startswith("CCE"):
        plt.ylabel("Exploitability", fontsize=fontsize)
    else:
        plt.ylabel("Regret", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    os.makedirs("./figures", exist_ok=True)
    plt.savefig('./figures/' + store_name + '.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    param = Parameter()
    para_list = param.param_list

    rounds = len(param.param['exp_idx'])
    full_param_list = [(_, rounds) for _ in para_list]

    algo_game = ["CCE_Instant", "CCE_Average"]
    algo_list = ["Instant", "Average", "CCE_Instant", "CCE_Average"]
    pool = Pool(processes=37)
    regret_mean = pool.starmap(draw_regret, full_param_list)
    keys_of_game = ["K", "tau", "b", "is_full_info"]
    best_regret = {_: {} for _ in algo_list}
    best_para = {_: {} for _ in algo_list}
    for i, para in enumerate(para_list):
        key_combination = tuple(para[key] for key in keys_of_game)
        algo = para["algo"]
        if key_combination not in best_regret[algo]:
            best_regret[algo][key_combination] = float('inf')
            best_para[algo][key_combination] = None
    game_para_default = [5, 1, 1.0, True]
    game_para_choice = {
        'K': [3, 5, 10],
        'tau': [0.5, 1, 2],
        'b': [0.3, 0.5, 0.7],
    }
    special_para_list = {
        'b': [0.1, 0.2, 0.3]
    }
    search_para_map = {
        'K': 0,
        'tau': 1,
        'b': 2,
        'is_full_info': 3
    }
    search_list_of_algo_fullinfo = {
        "Instant": ['tau', "b"],
        "Average": ['tau'],
        "CCE_Instant": ['tau'],
        "CCE_Average": ['tau']
    }
    search_list_of_algo_bandit = {
        "Instant": ["K", 'tau', "b"],
        "Average": ['K', 'tau', 'b'],
        "CCE_Instant": ["K", 'tau'],
        "CCE_Average": ['K', 'tau']
    }
    for i, para in enumerate(para_list):
        key_combination = tuple(para[key] for key in keys_of_game)
        regret_mean_now = regret_mean[i]
        algo = para["algo"]
        if regret_mean_now < best_regret[algo][key_combination]:
            best_regret[algo][key_combination] = regret_mean_now
            best_para[algo][key_combination] = para
    color_list = ["#FF7F00", "#4DAF4A", "#F781BF"]
    linestyle_list = ["-", "--", "-."]

    # for fullinfo
    for algo in algo_list:
        print("algo: ", algo)
        for search_now in search_list_of_algo_fullinfo[algo]:
            print("search_now: ", search_now)
            paras_search_now = game_para_choice[search_now]
            para_to_draw = []
            for para_used_now in paras_search_now:
                try_para = copy.deepcopy(game_para_default)
                if algo == "Instant":
                    try_para[2] = 0.7
                try_para[0] = 10
                try_para[search_para_map[search_now]] = para_used_now
                try_para = tuple(try_para)
                print("try_para: ", try_para)
                para_to_draw.append(best_para[algo][try_para])
            draw_multiple_regrets(para_to_draw, search_now, rounds, color_list, linestyle_list)

    # for bandit
    for algo in algo_list:
        print("algo: ", algo)
        for search_now in search_list_of_algo_bandit[algo]:
            print("search_now: ", search_now)
            paras_search_now = game_para_choice[search_now]
            if algo == 'Average' and search_now == 'b':
                paras_search_now = special_para_list[search_now]
            para_to_draw = []
            for para_used_now in paras_search_now:
                try_para = copy.deepcopy(game_para_default)
                try_para[3] = False
                if algo == "Instant":
                    try_para[2] = 0.5
                if algo == 'Average':
                    try_para[2] = 0.2
                try_para[search_para_map[search_now]] = para_used_now
                try_para = tuple(try_para)
                print("try_para: ", try_para)
                para_to_draw.append(best_para[algo][try_para])
            draw_multiple_regrets(para_to_draw, search_now, rounds, color_list, linestyle_list)
    exit(0)
