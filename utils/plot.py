# nbit-5, with truncation, one figure, all T.
# with smooth

# 25th/January/2021: T fixed. nbit-5 fixed.

import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
plt.switch_backend('agg')

total_steps = 2000000

title_name = "HalfCheetah-v3"
original_path = "/home/pami/Desktop/Continuous_Algo/results/HalfCheetah-v3/*"
smooth = True
smooth_index = 2 # [1, +max), could be changed into 1
threshold_len = 10000 # == 10000? if (threshold_len>10000): threshold <- 10000
linewidth = 1.5 # linewidth of the figure line
scatter_space = 10 # mark space out each (scatter_space) point
key_name = "Test_Episode_Reward"
# FF7100 orange
color_dictionary = {
    0: "#00AB2C", 1: "#A055BE", 2: "#FF7100", 3: "#F45CC1", 4: "#945047",
    5: "#34A8E3", 6: "#800000", 7: "#008000", 8: "#008000", 9: "#E74C3C",
    10: "#D35400", 11: "#800000", 12: "#0E0F0F", 13: "#F1948A", 14: "#1C2833",
    15: "#F322CD", 16: "#1F618D"
}

marker_dictionary = {
    0:"o", 1:"^", 2:"D", 3:"x", 4:"+", 5:"*"
}
# linestyle_dictionary = {
#     0:"-", 1:"--", 2:":", 3:"-."
# }

def mean_std_fillcolor_plot(thresholds, color, label, marker):
    thresholds_mean = thresholds.mean(axis = 0)
    print("thresholds_mean: ", thresholds_mean)

    x = [i for i in range(len(thresholds_mean))]
    thresholds_std = thresholds.std(axis = 0)
    superbound = thresholds_mean + thresholds_std
    lowerbound = thresholds_mean - thresholds_std

    x_scatter = np.zeros(0)
    thresholds_mean_scatter = np.zeros(0)
    print("x_scatter: ", x_scatter)
    scatter_index = 0
    x_len = len(x)

    # changed here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x = np.arange(1, total_steps, total_steps/x_len) # Ant: x_len == 90
    while(True):
        x_scatter = np.concatenate((x_scatter, [x[scatter_index]]), axis = 0)
        thresholds_mean_scatter = np.concatenate((thresholds_mean_scatter, [thresholds_mean[scatter_index]]), axis = 0)
        scatter_index += scatter_space
        if (scatter_index >= x_len):
            break
    print("x_scatter: ", x_scatter)

    plt.plot(x, thresholds_mean, color=color, linewidth=linewidth) 
    plt.scatter(x_scatter, thresholds_mean_scatter, color=color, label=label, marker=marker)
    plt.fill_between(x, superbound, lowerbound, where=superbound>=lowerbound, facecolor=color, interpolate=True, alpha=0.20)
    return

def smooth_func(threshold, tmp_threshold):
    for i in range(len(threshold)): # average with the behind
        # threshold.value = threshold[i:i+smooth_index].value
        # print("(threshold[i].step, threshold[i].value): ", (threshold[i].step, threshold[i].value))
        tmp_threshold[i] = (np.array([j.value for j in threshold[i:i+smooth_index]])).mean()
    return tmp_threshold

def plot_func(dirs, color, label, marker):
    threshold_list = list()
    for dir in dirs:
        print("current dir: ", dir)
        event = EventAccumulator(dir)
        event.Reload()
        print("event.scalars.Keys(): ", event.scalars.Keys())
        # threshold 
        if (label == "2m_original" or label == "original"):
            label = "SAC"
            threshold = event.scalars.Items(key_name)
        elif (label == "2m_implicit_112"):
            label = "SAC + Selective Sampling"
            threshold = event.scalars.Items(key_name)
        elif (label == "2m_implicit_211"):
            label = "SAC + Selective Sampling"
            threshold = event.scalars.Items(key_name)
        elif (label == "2m_implicit_004"):
            threshold = event.scalars.Items(key_name)
        else:
            threshold = event.scalars.Items(key_name)

        threshold_len = len(threshold)
        tmp_threshold = np.zeros(threshold_len)
        if (not smooth):
            smooth_index = 1
        tmp_threshold = smooth_func(threshold, tmp_threshold)
        threshold_list.append(tmp_threshold)

    # minimum threshold length of all thresholds from one T and different seeds: 
    min_threshold_len = min([len(threshold_list[i]) for i in range(len(threshold_list))])
    thresholds = np.zeros(0)
    for i in range(0, len(threshold_list)):
        thresholds = np.concatenate((thresholds, threshold_list[i][0:min_threshold_len]), axis = 0)
    thresholds = thresholds.reshape((len(threshold_list), min_threshold_len))
    mean_std_fillcolor_plot(thresholds, color, label, marker)
    # plt.plot([i.step for i in threshold], tmp_threshold, color=color, label=label, linewidth=linewidth) 

def dirs_process():
    # dir func:
    method_paths = glob.glob(original_path) # paths type:string list
    methodname_list = list()

    for i in range(len(method_paths)):
        methodname_list.append(os.path.basename(method_paths[i]))
    # print("methodname_list: ", methodname_list)
    # print("method_paths: ", method_paths)

    final_path = list()
    for i in range(len(method_paths)):
        final_path.append(glob.glob(method_paths[i] + "/*"))
    print("final_path: ", final_path)

    for i in range(len(final_path)):
        plot_func(final_path[i], color_dictionary[i], methodname_list[i], marker_dictionary[i])

def main():
    dirs_process()


    # plt.figure(facecolor='blue',    # 图表区的背景色
    #            edgecolor='black')    # 图表区的边框线颜色
    ax=plt.gca()
    ax.patch.set_facecolor("#EAE9F3")    # 设置 ax1 区域背景颜色    
    ax.patch.set_alpha(0.8)    # 设置 ax1 区域背景颜色透明度  
    ax.set_ylim(0, 9000)
    ax.set_xlim(0, total_steps)
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    plt.grid(True)
    ax.grid(color='white',
            linestyle='-',
            linewidth=1,
            alpha=0.8)

    plt.xlabel("step")
    plt.ylabel("test episode reward")

    plt.legend(loc='lower right')
    # plt.legend(bbox_to_anchor=(0.4, -0.3), loc="lower center")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)

    plt.title(title_name)
    plt.savefig('/home/pami/Desktop/Continuous_Algo/figures/original.pdf', dpi=600, bbox_inches='tight')

    plt.xlim(0, 1600000)
    plt.savefig('/home/pami/Desktop/Continuous_Algo/figures/truncation_x.pdf', dpi=600, bbox_inches='tight')

    plt.xlim(0, 1500000)
    plt.savefig('/home/pami/Desktop/Continuous_Algo/figures/truncation_x2.pdf', dpi=600, bbox_inches='tight')

    plt.xlim(0, 1400000)
    plt.savefig('/home/pami/Desktop/Continuous_Algo/figures/truncation_x3.pdf', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    main() 
