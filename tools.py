import copy
import random
import math
import time
import os
import platform
import pickle
import scikit_posthocs
import shutil
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
import warnings
import sklearn
import json

from shutil import copyfile
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import pairwise_distances

from strings import CLINICAL_LABELS, DATA_SETS
from box_adjusted import draw_boxplt


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)
    y_mb   = y[idx].astype(float)

    return x_mb, y_mb


def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0:  # no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def string_to_stamp(string, string_format="%Y%m%d"):
    string = str(string)
    return time.mktime(time.strptime(string, string_format))


def minus_to_month(str1, str2):
    return (string_to_stamp(str2) - string_to_stamp(str1)) / 86400 / 30


def load_data(main_path, file_name):
    return np.load(main_path + file_name, allow_pickle=True)


def draw_heat_map(data, s=2):
    data = np.asarray(data)
    data_normed = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    data_normed = data_normed / s
    xlabels = ["MMSE", "CDRSB", "ADAS13"]
    ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    plt.figure()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data_normed, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title('DPS-Net')
    plt.show()


def draw_320(data, k=5):
    data_clear = []
    for item in data:
        data_clear.append(list(item) + [np.nan] * (9 - len(item)))
    print(data_clear)
    data_clear = np.asarray(data_clear).swapaxes(0, 1)
    plt.figure(dpi=400, figsize=(60, 3))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(data_clear, interpolation='nearest', cmap=plt.cm.jet, vmin=0, vmax=(k-1))
    plt.title('320*9')
    plt.savefig("test/pic1.png", bbox_inches="tight")
    plt.show()


def draw_heat_map_2(data1, data2, save_path, s=2, show_flag=False):
    pic_keys = ["var", "avg"]
    color_dic = {
        "var": plt.cm.hot,
        "avg": plt.cm.cool
    }
    label_dic = {
        "var": "variance",
        "avg": "average"
    }
    k = len(data1)
    for one_key in pic_keys:
        data1_tmp = [item.get(one_key) for item in copy.deepcopy(data1)]
        data2_tmp = [item.get(one_key) for item in copy.deepcopy(data2)]
        data_all = np.vstack((data1_tmp, data2_tmp))
        # print(data_all.shape)
        # print("1", data_all)
        data_all_trans = data_all.swapaxes(0, 1)
        new_lines = []
        for line in data_all_trans:
            new_lines.append(np.abs((line - np.nanmean(line)) / np.nanstd(line)))
        data_all = np.asarray(new_lines).swapaxes(0, 1)
        #data_all = np.abs((data_all - np.nanmean(data_all, axis=0) / np.nanstd(data_all, axis=0)))
        # print("2", data_all.shape, data_all)

        data_all = data_all / s
        data1_normed = data_all[:len(data1)]
        data2_normed = data_all[-len(data2):]
        data1_normed = np.nan_to_num(data1_normed, nan=10)
        data2_normed = np.nan_to_num(data2_normed, nan=10)
        # print("3", data1_normed)
        # print("4", data2_normed)
        # data1 = np.asarray(data1)
        # data1_normed = np.abs((data1 - data1.mean(axis=0)) / data1.std(axis=0))
        # data1_normed = data1_normed / s
        # data2 = np.asarray(data2)
        # data2_normed = np.abs((data2 - data2.mean(axis=0)) / data2.std(axis=0))
        # data2_normed = data2_normed / s
        xlabels = CLINICAL_LABELS
        ylabels = ["Subtype #{0}".format(i) for i in range(1, k + 1)]
        fig = plt.figure(dpi=300, figsize=(21, 9))
        ax = fig.add_subplot(121)
        ax.set_title("k-means")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data1_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})
        ax = fig.add_subplot(122)
        ax.set_title("DPS-Net")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data2_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})
        plt.tight_layout()
        plt.savefig("{}_{}.png".format(save_path, one_key), dpi=300)
        if show_flag:
            plt.show()


def draw_heat_map_3(data1, data2, data3, save_path, s=2, show_flag=False, filter_cols=7):
    pic_keys = ["var", "avg"]
    color_dic = {
        "var": plt.cm.hot,
        "avg": plt.cm.cool
    }
    label_dic = {
        "var": "variance",
        "avg": "average"
    }
    k = len(data1)
    for one_key in pic_keys:
        data1_tmp = [item.get(one_key) for item in copy.deepcopy(data1)]
        data2_tmp = [item.get(one_key) for item in copy.deepcopy(data2)]
        data3_tmp = [item.get(one_key) for item in copy.deepcopy(data3)]
        data_all = np.vstack((data1_tmp, data2_tmp, data3_tmp))
        data_all = data_all[:, -filter_cols:]
        # print(data_all.shape)
        # print("1", data_all)
        data_all_trans = data_all.swapaxes(0, 1)
        new_lines = []
        for line in data_all_trans:
            new_lines.append(np.abs((line - np.nanmean(line)) / np.nanstd(line)))
        data_all = np.asarray(new_lines).swapaxes(0, 1)
        #data_all = np.abs((data_all - np.nanmean(data_all, axis=0) / np.nanstd(data_all, axis=0)))
        # print("2", data_all.shape, data_all)

        data_all = data_all / s
        data1_normed = data_all[: len(data1)]
        data2_normed = data_all[len(data1): len(data1) + len(data2)]
        data3_normed = data_all[len(data1) + len(data2):]

        # data1_normed = np.nan_to_num(data1_normed, nan=10)
        # data2_normed = np.nan_to_num(data2_normed, nan=10)

        # print("3", data1_normed)
        # print("4", data2_normed)
        # data1 = np.asarray(data1)
        # data1_normed = np.abs((data1 - data1.mean(axis=0)) / data1.std(axis=0))
        # data1_normed = data1_normed / s
        # data2 = np.asarray(data2)
        # data2_normed = np.abs((data2 - data2.mean(axis=0)) / data2.std(axis=0))
        # data2_normed = data2_normed / s
        xlabels = CLINICAL_LABELS[-filter_cols:]
        ylabels = ["Subtype #{0}".format(i) for i in range(1, k + 1)]
        fig_size_dic = {
            7: (16, 9),
            14: (30, 9)
        }
        fig = plt.figure(dpi=300, figsize=fig_size_dic[filter_cols])
        ax = fig.add_subplot(131)
        ax.set_title("K-means")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data1_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        ax = fig.add_subplot(132)
        ax.set_title("SuStaIn")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data2_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        ax = fig.add_subplot(133)
        ax.set_title("DPS-Net")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data3_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        plt.tight_layout()
        plt.savefig("{}_{}.png".format(save_path, one_key), dpi=300)
        if show_flag:
            plt.show()


def draw_heat_map_4(data1, data2, data3, data4, save_path, s=2, show_flag=False):
    pic_keys = ["var", "avg"]
    color_dic = {
        "var": plt.cm.hot,
        "avg": plt.cm.cool
    }
    label_dic = {
        "var": "variance",
        "avg": "average"
    }
    k = len(data1)
    for one_key in pic_keys:
        data1_tmp = [item.get(one_key) for item in copy.deepcopy(data1)]
        data2_tmp = [item.get(one_key) for item in copy.deepcopy(data2)]
        data3_tmp = [item.get(one_key) for item in copy.deepcopy(data3)]
        data4_tmp = [item.get(one_key) for item in copy.deepcopy(data4)]
        data_all = np.vstack((data1_tmp, data2_tmp, data3_tmp, data4_tmp))
        data_all_trans = data_all.swapaxes(0, 1)
        new_lines = []
        for line in data_all_trans:
            new_lines.append(np.abs((line - np.nanmean(line)) / np.nanstd(line)))
        data_all = np.asarray(new_lines).swapaxes(0, 1)
        data_all = data_all / s
        data1_normed = data_all[: len(data1)]
        data2_normed = data_all[len(data1): len(data1) + len(data2)]
        data3_normed = data_all[len(data1) + len(data2): len(data1) + len(data2) + len(data3)]
        data4_normed = data_all[len(data1) + len(data2) + len(data3):]
        xlabels = CLINICAL_LABELS
        ylabels = ["Subtype #{0}".format(i) for i in range(1, k + 1)]
        fig = plt.figure(dpi=300, figsize=(16, 9))

        ax = fig.add_subplot(221)
        ax.set_title("K-means")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data1_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        ax = fig.add_subplot(222)
        ax.set_title("SuStaIn")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data2_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        ax = fig.add_subplot(223)
        ax.set_title("DTC")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data3_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        ax = fig.add_subplot(224)
        ax.set_title("DPS-Net")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        im = ax.imshow(data4_normed, cmap=color_dic[one_key], vmin=0, vmax=1)
        cb = plt.colorbar(im, shrink=0.4)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Intra-cluster {}".format(label_dic[one_key]), fontdict={"rotation": 270})

        plt.tight_layout()
        plt.savefig("{}_{}.png".format(save_path, one_key), dpi=300)
        if show_flag:
            plt.show()


def draw_stairs_2(data1, data2, save_path, threshold=0.05):
    # print("in draw_stairs:")
    # print("data1:")
    # print(data1)
    # print("data2:")
    # print(data2)
    k = len(data1)
    for i, one_target in enumerate(CLINICAL_LABELS):
        k1 = np.asarray(copy.deepcopy(data1[i]))
        k2 = np.asarray(copy.deepcopy(data2[i]))
        for j in range(len(k1)):
            for l in range(len(k1[j])):
                if k1[j][l] <= threshold:
                    k1[j][l] = 1
                else:
                    k1[j][l] = np.nan
        for j in range(len(k2)):
            for l in range(len(k2[j])):
                if k2[j][l] <= threshold:
                    k2[j][l] = 1
                else:
                    k2[j][l] = np.nan
        ylabels = range(2, k + 1)  # [2, 3, 4, 5]
        xlabels = range(1, k)  # [1, 2, 3, 4]
        fig = plt.figure(dpi=400, figsize=(7, 3))

        ax = fig.add_subplot(121)
        ax.set_title("K-means")
        ax.set_xticks(np.arange(0, k - 1, 1))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(0, k - 1, 1))
        ax.set_yticklabels(ylabels)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        # for one_line in [
        #     [-0.5, -0.5, 3.5],
        #     [0.5, -0.5, 3.5],
        #     [1.5, 0.5, 3.5],
        #     [2.5, 1.5, 3.5],
        #     [3.495, 2.5, 3.5]
        # ]:
        #     ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        # for one_line in [
        #     [-0.495, -0.5, 0.5],
        #     [0.5, -0.5, 1.5],
        #     [1.5, -0.5, 2.5],
        #     [2.5, -0.5, 3.5],
        #     [3.5, -0.5, 3.5]
        # ]:
        #     ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        ax.imshow(k1, cmap=plt.cm.Reds, vmin=0, vmax=1.5)

        ax = fig.add_subplot(122)
        ax.set_title("DPS-Net")
        ax.set_xticks(np.arange(0, k - 1, 1))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(0, k - 1, 1))
        ax.set_yticklabels(ylabels)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        # for one_line in [
        #     [-0.5, -0.5, 3.5],
        #     [0.5, -0.5, 3.5],
        #     [1.5, 0.5, 3.5],
        #     [2.5, 1.5, 3.5],
        #     [3.495, 2.5, 3.5]]:
        #     ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        # for one_line in [
        #     [-0.495, -0.5, 0.5],
        #     [0.5, -0.5, 1.5],
        #     [1.5, -0.5, 2.5],
        #     [2.5, -0.5, 3.5],
        #     [3.5, -0.5, 3.5]]:
        #     ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        ax.imshow(k2, cmap=plt.cm.Reds, vmin=0, vmax=1.5)
        plt.suptitle("Inter-cluster difference [{}]".format(one_target))
        plt.tight_layout()
        plt.savefig("{}_{}.png".format(save_path, one_target), dpi=400)
        # plt.show()


def draw_stairs_3(data1, data2, data3, save_path, threshold=0.05, flag=False, show_flag=False):
    k = np.asarray(data1).shape[1] + 1
    # print(data1)
    # print(np.asarray(data1).shape)
    stair_labels = CLINICAL_LABELS
    if flag:
        stair_labels = ["MMSE", "CDRSB", "ADAS"]
    for i, one_target in enumerate(stair_labels):
        k1 = np.asarray(copy.deepcopy(data1[i]))
        k2 = np.asarray(copy.deepcopy(data2[i]))
        k3 = np.asarray(copy.deepcopy(data3[i]))
        # print(k1.shape, k2.shape, k3.shape)
        k1_count = 0
        for j in range(len(k1)):
            for l in range(len(k1[j])):
                if k1[j][l] <= threshold:
                    k1[j][l] = 1
                    k1_count += 1
                else:
                    k1[j][l] = np.nan
        k2_count = 0
        for j in range(len(k2)):
            for l in range(len(k2[j])):
                if k2[j][l] <= threshold:
                    k2[j][l] = 1
                    k2_count += 1
                else:
                    k2[j][l] = np.nan
        k3_count = 0
        for j in range(len(k3)):
            for l in range(len(k3[j])):
                if k3[j][l] <= threshold:
                    k3[j][l] = 1
                    k3_count += 1
                else:
                    k3[j][l] = np.nan
        print(k1_count, k2_count, k3_count)
        ylabels = range(2, k + 1)  # [2, 3, 4, 5]
        xlabels = range(1, k)  # [1, 2, 3, 4]
        v_lines = [[-0.5, -0.5, k - 1.5]] + [[0.5 + i, -0.5 + i, k - 1.5] for i in range(k - 2)] + [[k - 1.505, k - 2.5, k - 1.5]]
        h_lines = [[-0.495, -0.5, 0.5]] + [[0.5 + i, -0.5, 1.5 + i] for i in range(k - 2)] + [[k - 1.5, -0.5, k - 1.5]]
        fig = plt.figure(dpi=300, figsize=(10, 3))

        ax = fig.add_subplot(131)
        ax.set_title("K-means")
        ax.set_xticks(np.arange(0, k - 1, 1))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(0, k - 1, 1))
        ax.set_yticklabels(ylabels)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        for one_line in v_lines:
            ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        for one_line in h_lines:
            ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        ax.imshow(k1, cmap=plt.cm.Reds, vmin=0, vmax=1.5)

        ax = fig.add_subplot(132)
        ax.set_title("SuStaIn")
        ax.set_xticks(np.arange(0, k - 1, 1))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(0, k - 1, 1))
        ax.set_yticklabels(ylabels)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        for one_line in v_lines:
            ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        for one_line in h_lines:
            ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        ax.imshow(k2, cmap=plt.cm.Reds, vmin=0, vmax=1.5)

        ax = fig.add_subplot(133)
        ax.set_title("MSSN")
        ax.set_xticks(np.arange(0, k - 1, 1))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(0, k - 1, 1))
        ax.set_yticklabels(ylabels)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        for one_line in v_lines:
            ax.vlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        for one_line in h_lines:
            ax.hlines(one_line[0], one_line[1], one_line[2], colors="black", linestyle="dotted", linewidth=1)
        ax.imshow(k3, cmap=plt.cm.Reds, vmin=0, vmax=1.5)

        plt.suptitle("Inter-cluster difference [{}]".format(one_target))
        plt.tight_layout()
        plt.savefig("{}_{}.png".format(save_path, one_target), dpi=400)
        if show_flag:
            plt.show()
        plt.clf()
        # break


def get_engine():
    if platform.system().lower() == "linux":
        return "openpyxl"
    elif platform.system().lower() == "windows":
        return None
    elif platform.system().lower() == "darwin":
        return "openpyxl"
    return None


def fill_nan(clinic_list):
    mean = np.nanmean(np.asarray(clinic_list))
    return [item if not math.isnan(item) else mean for item in clinic_list]


def get_heat_map_data(main_path, K, label, data_type):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)

    data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    target_labels = CLINICAL_LABELS
    data = data[["PTID", "EXAMDATE"] + target_labels]
    data = data[pd.notnull(data["EcogPtMem"])]

    for one_label in target_labels:
        data[one_label] = fill_nan(data[one_label])

    result = []

    for i in range(K):
        dic = dict()
        for one_target_label in target_labels:
            dic[one_target_label] = []
        for j, one_pt_id in enumerate(pt_ids):
            for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
                if label[j][k] == i:
                    for one_target_label in target_labels:
                        tmp = data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][one_target_label].values[0]
                        dic[one_target_label] += [float(tmp)]
        try:
            tmp_var = [np.var(np.asarray(dic[one_target_label])) for one_target_label in target_labels]
            tmp_avg = [np.mean(np.asarray(dic[one_target_label])) for one_target_label in target_labels]
        except Exception as e:
            print("Error in var and avg:", e)
            tmp_var = [np.nan for one_target_label in target_labels]
            tmp_avg = [np.nan for one_target_label in target_labels]
        result.append({
            "var": tmp_var,
            "avg": tmp_avg
        })

    return result


def make_heat_map_data_box(main_path, save_path, label, data_type):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)
    data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    target_labels = CLINICAL_LABELS
    data = data[["PTID", "EXAMDATE"] + target_labels]
    data = data[pd.notnull(data["EcogPtMem"])]
    for one_label in target_labels:
        data[one_label] = fill_nan(data[one_label])
    output_dic = dict()
    for i, one_pt_id in enumerate(pt_ids):
        for j, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            tmp_dic = dict()
            tmp_dic["label"] = label[i][j]
            tmp_dic["clinical"] = []
            for one_target_label in target_labels:
                tmp_clinical = data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][one_target_label].values[0]
                tmp_dic["clinical"] += [float(tmp_clinical)]
            output_dic["{}/{}".format(one_pt_id, one_exam_date)] = tmp_dic
    # print(len(list(output_dic.keys())), output_dic.keys())
    with open(save_path, "wb") as f:
        pickle.dump(output_dic, f)
    return


def one_time_tsne_data_x(main_path, label, data_name):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_name[:-1])
    data_x_raw = load_data(main_path, "/data/data_x/data_x_{}.npy".format(data_name))
    # data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    # target_labels = CLINICAL_LABELS
    # data = data[["PTID", "EXAMDATE"] + target_labels]
    # data = data[pd.notnull(data["EcogPtMem"])]
    #
    # for one_label in target_labels:
    #     data[one_label] = fill_nan(data[one_label])
    # output_data = [[] for i in range(K)]
    output_data = []
    colors = []
    for i, one_pt_id in enumerate(pt_ids):
        for j, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            colors.append(label[i][j])
            output_data.append(data_x_raw[i][j])
    # print([len(item) for item in output_data])
    with open("test/tsne_data.pkl", "wb") as f:
        pickle.dump(output_data, f)
    with open("test/tsne_data_colors.pkl", "wb") as f:
        pickle.dump(colors, f)
    return


def one_time_tsne_data_y(main_path, label, data_name):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_name[:-1])
    data_y = load_data(main_path, "/data/data_y/data_y_{}.npy".format(data_name[:-1]))
    # data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    # target_labels = CLINICAL_LABELS
    # data = data[["PTID", "EXAMDATE"] + target_labels]
    # data = data[pd.notnull(data["EcogPtMem"])]
    #
    # for one_label in target_labels:
    #     data[one_label] = fill_nan(data[one_label])
    # output_data = [[] for i in range(K)]
    target_labels = [""]
    output_data = []
    colors = []
    for i, one_pt_id in enumerate(pt_ids):
        for j, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            colors.append(label[i][j])
            output_data.append(data_y[i][j])
    # print([len(item) for item in output_data])
    with open("test/tsne_data.pkl", "wb") as f:
        pickle.dump(output_data, f)
    with open("test/tsne_data_colors.pkl", "wb") as f:
        pickle.dump(colors, f)
    return


def one_time_deal_tsne(perplexity=30):
    with open("test/tsne_data.pkl", "rb") as f:
        data = pickle.load(f)
    # lengths = [len(item) for item in data]
    # k = len(lengths)
    # print(lengths)
    # data_all = []
    # for item in data:
    #     data_all += item
    data = np.array(data)
    print(data.shape)
    t0 = time.time()
    data_all_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12, learning_rate=300, init="random").fit_transform(data)
    # data_all_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12, learning_rate=300,
    #                          init='random').fit_transform(data_all)
    t1 = time.time()
    print(t1 - t0, "s")
    print(data_all_embedded.shape)
    # data_plot = []
    # index = 0
    # for one_length in lengths:
    #     data_plot.append(data_all_embedded[0: one_length])
    #     index += one_length
    # print([len(item) for item in data_plot])
    with open("test/tsne_data_plot.pkl", "wb") as f:
        pickle.dump(data_all_embedded, f)


def one_time_draw_tsne():
    with open("test/tsne_data_plot.pkl", "rb") as f:
        data = pickle.load(f)
    with open("test/tsne_data_colors.pkl", "rb") as f:
        colors = pickle.load(f)
    print([colors.count(item) for item in range(len(list(set(colors))))])
    # print([len(item) for item in data_plot])
    color_types = ["red", "cyan", "blue", "green", "orange", "yellow", "magenta"]
    for i, point in enumerate(data):
        plt.scatter(point[0], point[1], s=5, c=color_types[colors[i]])
    plt.show()


def get_heat_map_data_inter(main_path, K, label, data_type, flag=False):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)

    data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    target_labels = CLINICAL_LABELS
    data = data[["PTID", "EXAMDATE"] + target_labels + ["MMSE", "CDRSB", "ADAS13"]]
    data = data[pd.notnull(data["EcogPtMem"])]



    for one_label in (target_labels + ["MMSE", "CDRSB", "ADAS13"]):
        data[one_label] = fill_nan(data[one_label])

    # scores = data[(target_labels + ["MMSE", "CDRSB", "ADAS13"])]
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaledX = scaler.fit_transform(scores)
    # np.set_printoptions(precision=3)  # Setting precision for the output
    # scores = rescaledX
    # scores = pd.DataFrame(scores)
    # scores.columns = target_labels + ["MMSE", "CDRSB", "ADAS13"]
    # for one_target_label in target_labels + ["MMSE", "CDRSB", "ADAS13"]:
    #     data[one_target_label] = scores[one_target_label]

    label_match = []
    for j, one_pt_id in enumerate(pt_ids):
        for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            label_match.append(label[j][k])
    matrix = []
    x_inter = []
    dt = []
    result = []

    for j, one_pt_id in enumerate(pt_ids):
        for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            tmp = []
            if flag:
                target_labels = ["MMSE", "CDRSB", "ADAS13"]  # ["CDRSB", "ADAS13"]
            for one_target_label in target_labels:
                tmp.append(float(data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][one_target_label].values[0]))
            x_inter.append(tmp)

    bad_result = np.asarray([np.nan] * (len(CLINICAL_LABELS) * (K - 1) ** 2)).reshape((len(CLINICAL_LABELS), K - 1, K - 1))
    print("x_inter=\n", x_inter)
    for j in range(K):
        tmp = []
        for i in range(len(label_match)):
            if label_match[i] == j:
                tmp.append(x_inter[i])
        if len(tmp) == 0:
            print("bad in drawing inter_cluster map")
            return 1, bad_result
        dt.append(tmp)
    print("dt=\n", dt)
    # with open("dt_gamma2_kmeans_seed_12.pkl", "wb") as f:
    #     pickle.dump(dt, f)
    for i in range(1, K):
        for j in range(0, K - 1):
            # if j > i:
            matrix.append(stats.ttest_ind(dt[i], dt[j])[1])
    print("matrix=\n", matrix)
    matrix = np.asarray(matrix).swapaxes(0, 1)
    for item in matrix:
        # tmp = item
        # for index in [1, 2, 3, 6, 7, 11]:
        #     tmp[index] = np.nan
        tmp = np.asarray(item).reshape(K - 1, K - 1)
        for i in range(len(tmp)):
            for j in range(i + 1, len(tmp)):
                tmp[i][j] = np.nan
        result.append(tmp)
    result = np.asarray(result)
    return 0, result


def count_inter(data_inter, threshold=0.05):
    dic = dict()
    # print("heat_map_data_inter in count_inter:")
    # print(data_inter)
    for i, one_target in enumerate(CLINICAL_LABELS):
        dic[one_target + "_inter_count"] = 0
        for j in range(len(data_inter[i])):
            for k in range(len(data_inter[i][j])):
                if data_inter[i][j][k] <= threshold:
                    dic[one_target + "_inter_count"] += 1
    return dic


def judge_good_train(labels, data_type, heat_map_data, heat_map_data_inter, flag, base_dic, k):
    # print("heat_map_data_inter in judge_good_train:")
    # print(heat_map_data_inter)
    cn_ad_labels = np.load("data/cn_ad_labels_{}.npy".format(data_type), allow_pickle=True)
    dic = dict()
    for i in range(k):
        dic[i] = 0
    for row in labels:
        for item in row:
            dic[item if (type(item) == int or type(item) == np.int32) else item[0]] += 1
    distribution = np.asarray([dic.get(i) for i in range(k)])
    label_strings = create_label_string(labels, cn_ad_labels, k)
    distribution_string = "/".join(["{}({})".format(x, y) for x, y in zip(distribution, label_strings)])
    param_cluster_std = distribution.std()
    heat_map_data_var = [item.get("var") for item in heat_map_data]
    fourteen_sums = np.asarray(heat_map_data_var).sum(axis=0) # three_sums = np.asarray(heat_map_data).sum(axis=0)
    count_inter_dic = count_inter(heat_map_data_inter)
    param_dic = dict()
    param_dic["Cluster_std"] = param_cluster_std
    for i, one_label in enumerate(CLINICAL_LABELS):
        param_dic[one_label + "_var"] = fourteen_sums[i]
        param_dic[one_label + "_inter_count"] = count_inter_dic.get(one_label + "_inter_count")
    clinical_judge_labels = [item + "_var" for item in CLINICAL_LABELS]
    clinical_judge_labels_inter = [item + "_inter_count" for item in CLINICAL_LABELS]
    if flag:
        judge = 0
        for one_label in clinical_judge_labels:
            if np.isnan(param_dic.get(one_label)):
                judge = -1
                break
        if judge != -1:
            # if param_dic.get("Cluster_std") < base_dic.get("Cluster_std"):
            #     judge += 1

            for one_label in clinical_judge_labels:
                if param_dic.get(one_label) < base_dic.get(one_label):
                    judge += 1
            for one_label in clinical_judge_labels_inter:
                if param_dic.get(one_label) > base_dic.get(one_label):
                    judge += 1

    else:
        judge = -1
    return judge, param_dic, distribution_string


def save_record(main_path, opt, index, distribution_string, judge, judge_params, comments, data_name, params=None):
    with open(main_path + "record/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/record_{}.csv".format(data_name, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), data_name), "a") as f:
        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},".format(
            index,
            judge,
            distribution_string,
            judge_params.get("Cluster_std"),
            judge_params.get("EcogPtMem_var"),
            judge_params.get("EcogPtLang_var"),
            judge_params.get("EcogPtVisspat_var"),
            judge_params.get("EcogPtPlan_var"),
            judge_params.get("EcogPtOrgan_var"),
            judge_params.get("EcogPtDivatt_var"),
            judge_params.get("EcogPtTotal_var"),
            judge_params.get("EcogSPMem_var"),
            judge_params.get("EcogSPLang_var"),
            judge_params.get("EcogSPVisspat_var"),
            judge_params.get("EcogSPPlan_var"),
            judge_params.get("EcogSPOrgan_var"),
            judge_params.get("EcogSPDivatt_var"),
            judge_params.get("EcogSPTotal_var"),
            judge_params.get("EcogPtMem_inter_count"),
            judge_params.get("EcogPtLang_inter_count"),
            judge_params.get("EcogPtVisspat_inter_count"),
            judge_params.get("EcogPtPlan_inter_count"),
            judge_params.get("EcogPtOrgan_inter_count"),
            judge_params.get("EcogPtDivatt_inter_count"),
            judge_params.get("EcogPtTotal_inter_count"),
            judge_params.get("EcogSPMem_inter_count"),
            judge_params.get("EcogSPLang_inter_count"),
            judge_params.get("EcogSPVisspat_inter_count"),
            judge_params.get("EcogSPPlan_inter_count"),
            judge_params.get("EcogSPOrgan_inter_count"),
            judge_params.get("EcogSPDivatt_inter_count"),
            judge_params.get("EcogSPTotal_inter_count"),
            comments
        ))
        if not params:
            f.write("".join([","] * 21))
        else:
            f.write(",".join([str(params.get(one_key)) for one_key in list(params.keys())]))
        f.write("\n")


def build_kmeans_result(main_path, opt, kmeans_labels, data_name, k):
    # kmeans_labels = np.asarray(kmeans_labels)
    res = get_heat_map_data(main_path, k, kmeans_labels, data_name[:-1])
    _, res_inter = get_heat_map_data_inter(main_path, k, kmeans_labels, data_name[:-1])
    judge, judge_params, distribution_string = judge_good_train(kmeans_labels, data_name[:-1], res, res_inter, False, None, k)
    # print(judge, judge_params, distribution_string)
    save_record(main_path, opt, -1, distribution_string, -1, judge_params, "kmeans_base", data_name)
    return judge_params, res, res_inter


def get_start_index(main_path, opt):
    df = pd.read_csv(main_path + "record/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/record_{}.csv".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), opt.data))
    # print(list(df["Id"]))
    start_index = max([int(item) for item in list(df["Id"])]) + 1
    # print(start_index)
    start_index = max(start_index, 1)
    return start_index


def build_cn_ad_labels(main_path, data_type):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    cn_ad_labels = []
    for pt_id in pt_ids:  # [148*148，[label tuple]，VISCODE，patientID]
        tmp_labels = []
        for one_exam_date in pt_dic.get(pt_id):
            one_label = list(clinical_score[(clinical_score["PTID"] == pt_id) & (clinical_score["EXAMDATE"] == one_exam_date)]["DX"])[0]
            tmp_labels.append(name_label(one_label))
        cn_ad_labels.append(tmp_labels)
    # print(cn_ad_labels)
    np.save("data/cn_ad_labels_{}.npy".format(data_type), cn_ad_labels, allow_pickle=True)
    # return np.asarray(cn_ad_labels)


def create_label_string(cluster_labels, const_cn_ad_labels, k):
    dic_list = []
    for i in range(k):
        dic = dict()
        dic["AD"] = 0
        dic["CN"] = 0
        dic["Other"] = 0
        dic_list.append(dic)

    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels[i])):
            tmp_cluster_id = cluster_labels[i][j] if (type(cluster_labels[i][j]) == int or type(cluster_labels[i][j]) == np.int32) else int(cluster_labels[i][j][0])
            if const_cn_ad_labels[i][j] == "AD":
                dic_list[tmp_cluster_id]["AD"] += 1
            elif const_cn_ad_labels[i][j] == "CN":
                dic_list[tmp_cluster_id]["CN"] += 1
            else:
                dic_list[tmp_cluster_id]["Other"] += 1
    # for dic in dic_list:
    #     print(dic)
    return ["{}+{}".format(dic.get("CN"), dic.get("AD")) for dic in dic_list]


def initial_record(main_path, opt, data_x_raw, data_name, seed_count, k):
    if not os.path.exists(main_path + "record/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/record_{}.csv".format(data_name, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), data_name)):
        os.makedirs(main_path + "record/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/".format(data_name, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch)))
        copyfile(main_path + "record/record_0.csv", main_path + "record/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/record_{}.csv".format(data_name, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), data_name))
        clinical_judge_labels = ["Cluster_std"] + [item + "_var" for item in CLINICAL_LABELS] + [item + "_inter_count" for item in CLINICAL_LABELS]
        dic = dict()
        res_all = []
        res_all_inter = []
        for one_label in clinical_judge_labels:
            dic[one_label] = 0
        print("Building kmeans bases... Please wait...")
        for seed in tqdm(range(seed_count)):
            kmeans_labels = get_kmeans_base(data_x_raw, seed, k)
            tmp_params, res, res_inter = build_kmeans_result(main_path, opt, kmeans_labels, data_name, k)
            res_all.append(res)
            res_all_inter.append(res_inter)
            for one_label in clinical_judge_labels:
                dic[one_label] += tmp_params.get(one_label)
        for one_label in clinical_judge_labels:
            dic[one_label] = round(dic.get(one_label) / seed_count, 2) if seed_count > 0 else 0
        with open("data/initial/{}/base_dic.pkl".format(data_name), "wb") as f:
            pickle.dump(dic, f)
        save_record(main_path, opt, 0, "None", -1, dic, "kmeans_base_average", data_name)
        if seed_count > 0:
            np.save("data/initial/{}/base_res.npy".format(data_name), res_all, allow_pickle=True)
            np.save("data/initial/{}/base_res_inter.npy".format(data_name), res_all_inter, allow_pickle=True)
            return dic, res_all[0], res_all_inter[0]
        else:
            empty = [[[0] * 14] for i in range(k)]
            empty_inter = np.asarray([np.nan] * (len(CLINICAL_LABELS) * (k - 1) ** 2)).reshape((len(CLINICAL_LABELS), k - 1, k - 1))
            np.save("data/initial/{}/base_res.npy".format(data_name), empty, allow_pickle=True)
            np.save("data/initial/{}/base_res_inter.npy".format(data_name), empty_inter, allow_pickle=True)
            return dic, empty, empty_inter

    else:
        with open("data/initial/{}/base_dic.pkl".format(data_name), "rb") as f:
            dic = pickle.load(f)
        base_res_all = np.load("data/initial/{}/base_res.npy".format(data_name), allow_pickle=True)
        base_res_inter_all = np.load("data/initial/{}/base_res_inter.npy".format(data_name), allow_pickle=True)
        return dic, base_res_all[0], base_res_inter_all[0]


def name_label(label):
    if label in ["CN", "SMC", "EMCI"]:
        return "CN"
    elif label in ["LMCI", "AD"]:
        return "AD"
    else:
        return None


def build_patient_dictionary(main_path):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    dic = dict()
    # ['EcogPtMem','EcogPtLang','EcogPtVisspat','EcogPtPlan','EcogPtOrgan','EcogPtDivatt','EcogPtTotal','EcogSPMem','EcogSPLang','EcogSPVisspat','EcogSPPlan','EcogSPOrgan','EcogSPDivatt','EcogSPTotal']
    for one_pt_id in tqdm(pt_ids):
        dates = list(clinical_score[(clinical_score["PTID"] == one_pt_id) & (pd.notnull(clinical_score["EcogPtMem"]))]["EXAMDATE"])
        # dates = list(clinical_score[
        #                       (clinical_score["PTID"] == one_pt_id) &
        #                       (pd.notnull(clinical_score["EcogPtMem"])) &
        #                       (pd.notnull(clinical_score["EcogPtLang"])) &
        #                       (pd.notnull(clinical_score["EcogPtVisspat"])) &
        #                       (pd.notnull(clinical_score["EcogPtPlan"])) &
        #                       (pd.notnull(clinical_score["EcogPtOrgan"])) &
        #                       (pd.notnull(clinical_score["EcogPtDivatt"])) &
        #                       (pd.notnull(clinical_score["EcogPtTotal"])) &
        #                       (pd.notnull(clinical_score["EcogSPMem"])) &
        #                       (pd.notnull(clinical_score["EcogSPLang"])) &
        #                       (pd.notnull(clinical_score["EcogSPVisspat"])) &
        #                       (pd.notnull(clinical_score["EcogSPPlan"])) &
        #                       (pd.notnull(clinical_score["EcogSPOrgan"])) &
        #                       (pd.notnull(clinical_score["EcogSPDivatt"])) &
        #                       (pd.notnull(clinical_score["EcogSPTotal"]))
        #                       ]["EXAMDATE"])
        first_date = dates[0]
        last_date = dates[-1]
        dic[one_pt_id] = [first_date, last_date]
    with open(main_path + "data/patient_dictionary.pkl", "wb") as f:
        pickle.dump(dic, f)


def load_patient_dictionary(main_path, data_type):
    with open(main_path + "data/patient_dictionary_{}.pkl".format(data_type), "rb") as f:
        pt_dic = pickle.load(f)
    return pt_dic


def get_kmeans_base(data_x_raw, seed, k):
    data = []
    for item in data_x_raw:
        for vec in item:
            data.append(vec)
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
    kmeans_output = []
    tmp_index = 0
    for item in data_x_raw:
        # print(len(item))
        kmeans_output.append(kmeans.labels_[tmp_index: tmp_index + len(item)])
        tmp_index += len(item)
    # print("tmp_index:", tmp_index)
    # dim = len(data_x[0])
    # for i in range(len(data_x)):
    #     tmp = kmeans.labels_[i * dim: i * dim + dim]
    #     kmeans_output.append(tmp)
    # kmeans_output = np.asarray(kmeans_output)
    return kmeans_output


def build_data_x_alpha(main_path):
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))

    data_x = np.load("data/data_x_new.npy", allow_pickle=True)
    data_x_alpha1 = data_x
    data_x_alpha2 = []
    data_x_alpha3 = []
    data_x_alpha4 = []
    for i in range(len(data_x)):
        data_x_alpha2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_alpha3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_alpha4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_alpha2 = np.asarray(data_x_alpha2)
    data_x_alpha3 = np.asarray(data_x_alpha3)
    data_x_alpha4 = np.asarray(data_x_alpha4)
    print(data_x_alpha1.shape)
    print(data_x_alpha2.shape)
    print(data_x_alpha3.shape)
    print(data_x_alpha4.shape)
    np.save(main_path + "data/data_x/data_x_alpha1.npy", data_x_alpha1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha2.npy", data_x_alpha2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha3.npy", data_x_alpha3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_alpha4.npy", data_x_alpha4, allow_pickle=True)


def build_data_x_beta(main_path, period=500, every=10):
    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    data_x = np.asarray([item[0: period: every] for item in data_x])

    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)
    print(data_x_beta1.shape)
    print(data_x_beta2.shape)
    print(data_x_beta3.shape)
    print(data_x_beta4.shape)
    np.save(main_path + "data/data_x/data_x_beta1.npy", data_x_beta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta2.npy", data_x_beta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta3.npy", data_x_beta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_beta4.npy", data_x_beta4, allow_pickle=True)


def build_data_y_beta(main_path):
    data_y = np.load(main_path + "data/data_y_new.npy", allow_pickle=True)
    print(np.shape(data_y))
    data_y = np.asarray([[item[-1]] for item in data_y])
    print(np.shape(data_y))
    np.save(main_path + "data/data_y/data_y_beta.npy", data_y, allow_pickle=True)


def create_empty_folders_all(main_path):
    locations = ["record/", "data/initial/", "saves/"]
    for one_location in locations:
        for one_dataset in DATA_SETS:
            tmp_dir = main_path + one_location + one_dataset
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
                print("Created folder {} successfully".format(tmp_dir))
            else:
                print("Folder {} exists".format(tmp_dir))


def create_empty_folders(main_path, data_name):
    locations = ["record/", "data/initial/", "saves/"]
    for one_location in locations:
        tmp_dir = main_path + one_location + data_name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print("Created folder {} successfully".format(tmp_dir))
        else:
            print("Folder {} exists".format(tmp_dir))


def count_pt_id_patient_lines(main_path):
    df = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    dic = dict()
    for one_key in pt_ids:
        dic[one_key] = 0
    # (clinical_score["PTID"] == one_pt_id) & (pd.notnull(clinical_score["EcogPtMem"]))
    for index, row in df.iterrows():
        if row.get("PTID") in pt_ids and pd.notnull(row.get("EcogPtMem")):
            dic[row.get("PTID")] += 1
    print(dic)
    summary = dict()
    for one_key in dic.keys():
        if dic.get(one_key) not in summary:
            summary[dic.get(one_key)] = 1
        else:
            summary[dic.get(one_key)] += 1
    for one_key in sorted(list(summary.keys()), key=lambda x: x):
        print("{}: {}".format(one_key, summary.get(one_key)))


def minus_date(str1, str2):
    return abs((string_to_stamp(str2) - string_to_stamp(str1)) / 86400)


def split_periods(periods, start=0, end=499):
    if len(periods) == 1:
        return [end]
    periods = sorted(periods, key=lambda x: x)
    periods = [str(item) for item in periods]
    periods_proportion = [minus_date(periods[i], periods[i + 1]) for i in range(0, len(periods) - 1)]
    outputs = [0]
    tmp = 0
    periods_sum = sum(periods_proportion)
    for item in periods_proportion:
        tmp += (end - start) * item / periods_sum
        outputs.append(round(tmp))
    return outputs


def split_periods_delta(periods):
    periods = sorted(periods, key=lambda x: x)
    periods = [str(item) for item in periods]
    outputs = [0]
    periods_proportion = [minus_date(periods[0], periods[i]) for i in range(1, len(periods))]
    for item in periods_proportion:
        outputs.append(round(item / 365.25 * 100))
    return outputs


def build_data_x_y_gamma(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["MMSE", "CDRSB", "ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["MMSE", "CDRSB", "ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_gamma.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_gamma.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_gamma1 = []
    data_x_gamma2 = []
    data_x_gamma3 = []
    data_x_gamma4 = []
    data_x_gamma1_raw = []
    data_x_gamma2_raw = []
    data_x_gamma3_raw = []
    data_x_gamma4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods(dic_date[pt_id])
        # print(dic_x_index[pt_id])
        # gamma1
        data_x_gamma1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma1_tmp) < max_length:
            data_x_gamma1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_gamma1_tmp))]
        data_x_gamma1.append(data_x_gamma1_tmp)

        # gamma2
        data_x_gamma2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma2_tmp) < max_length:
            data_x_gamma2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_gamma2_tmp))]
        data_x_gamma2.append(data_x_gamma2_tmp)

        # gamma3
        data_x_gamma3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma3_tmp) < max_length:
            data_x_gamma3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_gamma3_tmp))]
        data_x_gamma3.append(data_x_gamma3_tmp)

        # gamma4
        data_x_gamma4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_gamma4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_gamma4_tmp) < max_length:
            data_x_gamma4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_gamma4_tmp))]
        data_x_gamma4.append(data_x_gamma4_tmp)

    data_x_gamma1 = np.asarray(data_x_gamma1)
    data_x_gamma2 = np.asarray(data_x_gamma2)
    data_x_gamma3 = np.asarray(data_x_gamma3)
    data_x_gamma4 = np.asarray(data_x_gamma4)

    print(data_x_gamma1.shape)
    print(data_x_gamma2.shape)
    print(data_x_gamma3.shape)
    print(data_x_gamma4.shape)
    np.save(main_path + "data/data_x/data_x_gamma1.npy", data_x_gamma1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma2.npy", data_x_gamma2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma3.npy", data_x_gamma3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma4.npy", data_x_gamma4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma1_raw.npy", data_x_gamma1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma2_raw.npy", data_x_gamma2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma3_raw.npy", data_x_gamma3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_gamma4_raw.npy", data_x_gamma4_raw, allow_pickle=True)


def build_data_x_y_delta(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["MMSE", "CDRSB", "ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["MMSE", "CDRSB", "ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_delta.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_delta.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_delta1 = []
    data_x_delta2 = []
    data_x_delta3 = []
    data_x_delta4 = []
    data_x_delta1_raw = []
    data_x_delta2_raw = []
    data_x_delta3_raw = []
    data_x_delta4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # delta1
        data_x_delta1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_delta1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_delta1_tmp) < max_length:
            data_x_delta1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_delta1_tmp))]
        data_x_delta1.append(data_x_delta1_tmp)

        # delta2
        data_x_delta2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_delta2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_delta2_tmp) < max_length:
            data_x_delta2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_delta2_tmp))]
        data_x_delta2.append(data_x_delta2_tmp)

        # delta3
        data_x_delta3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_delta3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_delta3_tmp) < max_length:
            data_x_delta3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_delta3_tmp))]
        data_x_delta3.append(data_x_delta3_tmp)

        # delta4
        data_x_delta4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_delta4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_delta4_tmp) < max_length:
            data_x_delta4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_delta4_tmp))]
        data_x_delta4.append(data_x_delta4_tmp)

    print(dic_x_index)

    data_x_delta1 = np.asarray(data_x_delta1)
    data_x_delta2 = np.asarray(data_x_delta2)
    data_x_delta3 = np.asarray(data_x_delta3)
    data_x_delta4 = np.asarray(data_x_delta4)

    print(data_x_delta1.shape)
    print(data_x_delta2.shape)
    print(data_x_delta3.shape)
    print(data_x_delta4.shape)
    np.save(main_path + "data/data_x/data_x_delta1.npy", data_x_delta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta2.npy", data_x_delta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta3.npy", data_x_delta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta4.npy", data_x_delta4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta1_raw.npy", data_x_delta1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta2_raw.npy", data_x_delta2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta3_raw.npy", data_x_delta3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_delta4_raw.npy", data_x_delta4_raw, allow_pickle=True)


def build_data_x_y_epsilon(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_epsilon.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_epsilon.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_epsilon1 = []
    data_x_epsilon2 = []
    data_x_epsilon3 = []
    data_x_epsilon4 = []
    data_x_epsilon1_raw = []
    data_x_epsilon2_raw = []
    data_x_epsilon3_raw = []
    data_x_epsilon4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # epsilon1
        data_x_epsilon1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_epsilon1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_epsilon1_tmp) < max_length:
            data_x_epsilon1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_epsilon1_tmp))]
        data_x_epsilon1.append(data_x_epsilon1_tmp)

        # epsilon2
        data_x_epsilon2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_epsilon2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_epsilon2_tmp) < max_length:
            data_x_epsilon2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_epsilon2_tmp))]
        data_x_epsilon2.append(data_x_epsilon2_tmp)

        # epsilon3
        data_x_epsilon3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_epsilon3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_epsilon3_tmp) < max_length:
            data_x_epsilon3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_epsilon3_tmp))]
        data_x_epsilon3.append(data_x_epsilon3_tmp)

        # epsilon4
        data_x_epsilon4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_epsilon4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_epsilon4_tmp) < max_length:
            data_x_epsilon4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_epsilon4_tmp))]
        data_x_epsilon4.append(data_x_epsilon4_tmp)

    print(dic_x_index)

    data_x_epsilon1 = np.asarray(data_x_epsilon1)
    data_x_epsilon2 = np.asarray(data_x_epsilon2)
    data_x_epsilon3 = np.asarray(data_x_epsilon3)
    data_x_epsilon4 = np.asarray(data_x_epsilon4)

    print(data_x_epsilon1.shape)
    print(data_x_epsilon2.shape)
    print(data_x_epsilon3.shape)
    print(data_x_epsilon4.shape)
    np.save(main_path + "data/data_x/data_x_epsilon1.npy", data_x_epsilon1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon2.npy", data_x_epsilon2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon3.npy", data_x_epsilon3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon4.npy", data_x_epsilon4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon1_raw.npy", data_x_epsilon1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon2_raw.npy", data_x_epsilon2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon3_raw.npy", data_x_epsilon3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_epsilon4_raw.npy", data_x_epsilon4_raw, allow_pickle=True)


def build_data_x_y_zeta(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal']
    df = df[["PTID", "EXAMDATE"] + target_labels]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal']
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_zeta.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_zeta.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_zeta1 = []
    data_x_zeta2 = []
    data_x_zeta3 = []
    data_x_zeta4 = []
    data_x_zeta1_raw = []
    data_x_zeta2_raw = []
    data_x_zeta3_raw = []
    data_x_zeta4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # zeta1
        data_x_zeta1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_zeta1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_zeta1_tmp) < max_length:
            data_x_zeta1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_zeta1_tmp))]
        data_x_zeta1.append(data_x_zeta1_tmp)

        # zeta2
        data_x_zeta2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_zeta2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_zeta2_tmp) < max_length:
            data_x_zeta2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_zeta2_tmp))]
        data_x_zeta2.append(data_x_zeta2_tmp)

        # zeta3
        data_x_zeta3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_zeta3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_zeta3_tmp) < max_length:
            data_x_zeta3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_zeta3_tmp))]
        data_x_zeta3.append(data_x_zeta3_tmp)

        # zeta4
        data_x_zeta4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_zeta4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_zeta4_tmp) < max_length:
            data_x_zeta4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_zeta4_tmp))]
        data_x_zeta4.append(data_x_zeta4_tmp)

    print(dic_x_index)

    data_x_zeta1 = np.asarray(data_x_zeta1)
    data_x_zeta2 = np.asarray(data_x_zeta2)
    data_x_zeta3 = np.asarray(data_x_zeta3)
    data_x_zeta4 = np.asarray(data_x_zeta4)

    print(data_x_zeta1.shape)
    print(data_x_zeta2.shape)
    print(data_x_zeta3.shape)
    print(data_x_zeta4.shape)
    np.save(main_path + "data/data_x/data_x_zeta1.npy", data_x_zeta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta2.npy", data_x_zeta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta3.npy", data_x_zeta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta4.npy", data_x_zeta4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta1_raw.npy", data_x_zeta1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta2_raw.npy", data_x_zeta2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta3_raw.npy", data_x_zeta3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_zeta4_raw.npy", data_x_zeta4_raw, allow_pickle=True)


def build_data_x_y_eta(main_path, max_length=9):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["CDRSB", "ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["CDRSB", "ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    with open(main_path + "data/patient_dictionary_eta.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y)
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_eta.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_eta1 = []
    data_x_eta2 = []
    data_x_eta3 = []
    data_x_eta4 = []
    data_x_eta1_raw = []
    data_x_eta2_raw = []
    data_x_eta3_raw = []
    data_x_eta4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # eta1
        data_x_eta1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_eta1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_eta1_tmp) < max_length:
            data_x_eta1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_eta1_tmp))]
        data_x_eta1.append(data_x_eta1_tmp)

        # eta2
        data_x_eta2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_eta2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_eta2_tmp) < max_length:
            data_x_eta2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_eta2_tmp))]
        data_x_eta2.append(data_x_eta2_tmp)

        # eta3
        data_x_eta3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_eta3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_eta3_tmp) < max_length:
            data_x_eta3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_eta3_tmp))]
        data_x_eta3.append(data_x_eta3_tmp)

        # eta4
        data_x_eta4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_eta4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_eta4_tmp) < max_length:
            data_x_eta4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_eta4_tmp))]
        data_x_eta4.append(data_x_eta4_tmp)

    print(dic_x_index)

    data_x_eta1 = np.asarray(data_x_eta1)
    data_x_eta2 = np.asarray(data_x_eta2)
    data_x_eta3 = np.asarray(data_x_eta3)
    data_x_eta4 = np.asarray(data_x_eta4)

    print(data_x_eta1.shape)
    print(data_x_eta2.shape)
    print(data_x_eta3.shape)
    print(data_x_eta4.shape)
    np.save(main_path + "data/data_x/data_x_eta1.npy", data_x_eta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta2.npy", data_x_eta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta3.npy", data_x_eta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta4.npy", data_x_eta4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta1_raw.npy", data_x_eta1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta2_raw.npy", data_x_eta2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta3_raw.npy", data_x_eta3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_eta4_raw.npy", data_x_eta4_raw, allow_pickle=True)


def build_data_x_y_theta(main_path, max_length=5):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["CDRSB", "ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["CDRSB", "ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    for pt_id in pt_ids:
        dic_date[pt_id] = dic_date[pt_id][:max_length]
    with open(main_path + "data/patient_dictionary_theta.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y[:max_length])
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("data_y.shape:", data_y.shape)
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_theta.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_theta1 = []
    data_x_theta2 = []
    data_x_theta3 = []
    data_x_theta4 = []
    data_x_theta1_raw = []
    data_x_theta2_raw = []
    data_x_theta3_raw = []
    data_x_theta4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # theta1
        data_x_theta1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_theta1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_theta1_tmp) < max_length:
            data_x_theta1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_theta1_tmp))]
        data_x_theta1.append(data_x_theta1_tmp[:max_length])

        # theta2
        data_x_theta2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_theta2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_theta2_tmp) < max_length:
            data_x_theta2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_theta2_tmp))]
        data_x_theta2.append(data_x_theta2_tmp[:max_length])

        # theta3
        data_x_theta3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_theta3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_theta3_tmp) < max_length:
            data_x_theta3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_theta3_tmp))]
        data_x_theta3.append(data_x_theta3_tmp[:max_length])

        # theta4
        data_x_theta4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_theta4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_theta4_tmp) < max_length:
            data_x_theta4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_theta4_tmp))]
        data_x_theta4.append(data_x_theta4_tmp[:max_length])

    print(dic_x_index)

    data_x_theta1 = np.asarray(data_x_theta1)
    data_x_theta2 = np.asarray(data_x_theta2)
    data_x_theta3 = np.asarray(data_x_theta3)
    data_x_theta4 = np.asarray(data_x_theta4)

    print(data_x_theta1.shape)
    print(data_x_theta2.shape)
    print(data_x_theta3.shape)
    print(data_x_theta4.shape)
    np.save(main_path + "data/data_x/data_x_theta1.npy", data_x_theta1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta2.npy", data_x_theta2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta3.npy", data_x_theta3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta4.npy", data_x_theta4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta1_raw.npy", data_x_theta1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta2_raw.npy", data_x_theta2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta3_raw.npy", data_x_theta3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_theta4_raw.npy", data_x_theta4_raw, allow_pickle=True)
    print(data_x_theta2)


def build_data_x_y_iota(main_path, max_length=1):
    df = pd.read_excel(main_path + "data/MRI_information_All_Measurement.xlsx", engine=get_engine())
    target_labels = ["CDRSB", "ADAS13"]
    df = df[["PTID", "EXAMDATE"] + target_labels + ["EcogPtMem"]]
    df = df[pd.notnull(df["EcogPtMem"])]
    # df["EXAMDATE"] = [str(item) for item in df["EXAMDATE"]]
    # df["PTID"] = [str(item) for item in df["PTID"]]
    scores = df[target_labels]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ["CDRSB", "ADAS13"]
    for one_target_label in target_labels:
        df[one_target_label] = scores[one_target_label]

    for one_label in target_labels:
        df[one_label] = fill_nan(df[one_label])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    dic_date = dict()

    dic_y = dict()
    for index, row in df.iterrows():
        pt_id = row.get("PTID")
        if pt_id in pt_ids:
            if pt_id not in dic_date:
                dic_date[pt_id] = [row.get("EXAMDATE")]
            else:
                dic_date[pt_id].append(row.get("EXAMDATE"))
            if pt_id not in dic_y:
                dic_y[pt_id] = [[row.get(one_target) for one_target in target_labels]]
            else:
                dic_y[pt_id].append([row.get(one_target) for one_target in target_labels])
    for pt_id in pt_ids:
        dic_date[pt_id] = dic_date[pt_id][:max_length]
    with open(main_path + "data/patient_dictionary_iota.pkl", "wb") as f:
        pickle.dump(dic_date, f)
    tmp_max = -1
    empty_count = 0
    data_y = []
    for pt_id in pt_ids:
        tmp_max = max(tmp_max, len(dic_y.get(pt_id)))
        tmp_y = dic_y.get(pt_id)
        empty_count += (max_length - len(tmp_y))
        if len(tmp_y) == 1:
            tmp_y.append(tmp_y[0])
        if len(tmp_y) < max_length:
            tmp_y += [[0] * len(target_labels) for i in range(max_length - len(tmp_y))]
        data_y.append(tmp_y[:max_length])
    data_y = np.asarray(data_y)
    data_y = np.reshape(data_y, (len(pt_ids), max_length, len(target_labels)))
    print("data_y.shape:", data_y.shape)
    print("tmp_max:", tmp_max)
    print("empty_count: {} / ({} * {}) = {}".format(empty_count, len(pt_ids), max_length, empty_count / (len(pt_ids) * max_length)))
    np.save(main_path + "data/data_y/data_y_iota.npy", data_y, allow_pickle=True)

    data_x = np.load(main_path + "data/pred_1500.npy", allow_pickle=True)
    # data_x = np.asarray([item[0: 500] for item in data_x])
    data_network = scio.loadmat(main_path + "data/network_centrality.mat")
    betweenness = np.abs(np.asarray([item[0] for item in data_network["betweenness"]]))
    closeness = np.abs(np.asarray([item[0] for item in data_network["closeness"]]))
    degree = np.abs(np.asarray([item[0] for item in data_network["degree"]]))
    laplacian = np.abs(np.asarray([item[0] for item in data_network["laplacian"]]))
    pagerank = np.abs(np.asarray([item[0] for item in data_network["pagerank"]]))
    data_x_beta1 = data_x
    data_x_beta2 = []
    data_x_beta3 = []
    data_x_beta4 = []
    for i in range(len(data_x)):
        data_x_beta2.append([np.concatenate((data_x[i][j], laplacian), axis=0) for j in range(len(data_x[0]))])
        data_x_beta3.append([np.concatenate((data_x[i][j], degree), axis=0) for j in range(len(data_x[0]))])
        data_x_beta4.append([np.concatenate((data_x[i][j], betweenness, closeness, degree, pagerank, laplacian), axis=0) for j in range(len(data_x[0]))])
    data_x_beta2 = np.asarray(data_x_beta2)
    data_x_beta3 = np.asarray(data_x_beta3)
    data_x_beta4 = np.asarray(data_x_beta4)

    dic_x_index = dict()
    data_x_iota1 = []
    data_x_iota2 = []
    data_x_iota3 = []
    data_x_iota4 = []
    data_x_iota1_raw = []
    data_x_iota2_raw = []
    data_x_iota3_raw = []
    data_x_iota4_raw = []
    for i, pt_id in enumerate(pt_ids):
        dic_x_index[pt_id] = split_periods_delta(dic_date[pt_id])

        # print(dic_x_index[pt_id])
        # iota1
        data_x_iota1_tmp = [list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]]
        data_x_iota1_raw.append([list(data_x_beta1[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_iota1_tmp) < max_length:
            data_x_iota1_tmp += [[0] * data_x_beta1.shape[2] for i in range(max_length - len(data_x_iota1_tmp))]
        data_x_iota1.append(data_x_iota1_tmp[:max_length])

        # iota2
        data_x_iota2_tmp = [list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]]
        data_x_iota2_raw.append([list(data_x_beta2[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_iota2_tmp) < max_length:
            data_x_iota2_tmp += [[0] * data_x_beta2.shape[2] for i in range(max_length - len(data_x_iota2_tmp))]
        data_x_iota2.append(data_x_iota2_tmp[:max_length])

        # iota3
        data_x_iota3_tmp = [list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]]
        data_x_iota3_raw.append([list(data_x_beta3[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_iota3_tmp) < max_length:
            data_x_iota3_tmp += [[0] * data_x_beta3.shape[2] for i in range(max_length - len(data_x_iota3_tmp))]
        data_x_iota3.append(data_x_iota3_tmp[:max_length])

        # iota4
        data_x_iota4_tmp = [list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]]
        data_x_iota4_raw.append([list(data_x_beta4[i][index]) for index in dic_x_index[pt_id]])
        if len(data_x_iota4_tmp) < max_length:
            data_x_iota4_tmp += [[0] * data_x_beta4.shape[2] for i in range(max_length - len(data_x_iota4_tmp))]
        data_x_iota4.append(data_x_iota4_tmp[:max_length])

    print(dic_x_index)

    data_x_iota1 = np.asarray(data_x_iota1)
    data_x_iota2 = np.asarray(data_x_iota2)
    data_x_iota3 = np.asarray(data_x_iota3)
    data_x_iota4 = np.asarray(data_x_iota4)

    print(data_x_iota1.shape)
    print(data_x_iota2.shape)
    print(data_x_iota3.shape)
    print(data_x_iota4.shape)
    np.save(main_path + "data/data_x/data_x_iota1.npy", data_x_iota1, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota2.npy", data_x_iota2, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota3.npy", data_x_iota3, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota4.npy", data_x_iota4, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota1_raw.npy", data_x_iota1_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota2_raw.npy", data_x_iota2_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota3_raw.npy", data_x_iota3_raw, allow_pickle=True)
    np.save(main_path + "data/data_x/data_x_iota4_raw.npy", data_x_iota4_raw, allow_pickle=True)
    print(data_x_iota2)


def one_time_draw_4(main_path):
    kmeans_label = np.load("test/kmeans_delta1_labels.npy", allow_pickle=True)
    sustain_label = np.load("test/sustain_delta1_labels_2.npy", allow_pickle=True)
    dtc_label = np.load("test/dtc_delta1_labels.npy", allow_pickle=True)
    dps_label = np.load("test/dps_delta1_labels_2.npy", allow_pickle=True)
    heat_map_data_kmeans = get_heat_map_data(main_path, 6, kmeans_label, "delta")
    heat_map_data_sustain = get_heat_map_data(main_path, 6, sustain_label, "delta")
    heat_map_data_dtc = get_heat_map_data(main_path, 6, dtc_label, "delta")
    heat_map_data_dps = get_heat_map_data(main_path, 6, dps_label, "delta")
    draw_heat_map_4(heat_map_data_kmeans, heat_map_data_sustain, heat_map_data_dtc, heat_map_data_dps, "test/comparison", 2, True)
    return


def one_time_draw_3(main_path):
    kmeans_label = np.load("test/kmeans_delta1_labels.npy", allow_pickle=True)
    sustain_label = np.load("test/sustain_delta1_labels_2.npy", allow_pickle=True)
    dps_label = np.load("test/dps_delta1_labels_2.npy", allow_pickle=True)
    heat_map_data_kmeans = get_heat_map_data(main_path, 6, kmeans_label, "delta")
    heat_map_data_sustain = get_heat_map_data(main_path, 6, sustain_label, "delta")
    heat_map_data_dps = get_heat_map_data(main_path, 6, dps_label, "delta")
    draw_heat_map_3(heat_map_data_kmeans, heat_map_data_sustain, heat_map_data_dps, "test/comparison", 2, True)
    return


def parse_flatten(main_path, flatten_labels, data_type):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_type)
    normal_labels = []
    index = 0
    for j, one_pt_id in enumerate(pt_ids):
        tmp_length = len(pt_dic.get(one_pt_id))
        normal_labels.append(list([int(item) for item in flatten_labels[index: index + tmp_length]]))
        index += tmp_length
        # for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
    return normal_labels


def one_time_label_trans(label, index=None, k=6, raw_flag=False):
    trans_table = [[0] * k for i in range(k)]
    trans_table_string = [[""] * (k + 1) for i in range(k)]
    count = [0] * k
    count_clear = [0] * k
    all_count = 0
    for item in label:
        length = len(item)
        all_count += length - 1
        for i in range(length - 1):
            trans_table[item[i]][item[i + 1]] += 1
        for i in range(length):
            count[item[i]] += 1
        for i in range(length - 1):
            count_clear[item[i]] += 1
    # print("all_count =", all_count)
    # print(trans_table)
    for i in range(k):
        tmp_sum = sum(trans_table[i])
        for j in range(k):
            trans_table_string[i][j] = "{1:.1f}%".format(trans_table[i][j], 100 * trans_table[i][j] / tmp_sum)
    # print(np.asarray(trans_table))
    order_list = [[trans_table[i][i], i] for i in range(k)]
    order_list.sort(key=lambda x: -x[0])
    if not raw_flag:
        order = [item[1] for item in order_list]
    else:
        order = range(0, k)
    #print(order)
    trans_table_ordered = [[""] * k for i in range(k)]
    for i in range(k):
        for j in range(k):
            trans_table_ordered[i][j] = trans_table_string[order[i]][order[j]]
    # print(np.asarray(trans_table_ordered))
    print("{}\tfrom\\to\tSubtype #1\tSubtype #2\tSubtype #3\tSubtype #4\tSubtype #5\tSubtype #6\tCount_label_clear\tCount_label".format(index))
    for i in range(k):
        print("\tSubtype #{}\t".format(i + 1), end="")
        for j in range(k):
            print("{0}\t".format(trans_table_ordered[i][j]), end="")
        print("{}(100.0%)\t{}".format(count_clear[order[i]], count[order[i]]), end="")
        print()


def one_time_label_trans_end_type(label, k=6):
    trans_table = [[0] * (k + 1) for i in range(k)]
    trans_table_string = [[""] * (k + 1) for i in range(k)]
    count = [0] * k
    # count_clear = [0] * k
    all_count = 0
    for item in label:
        length = len(item)
        all_count += length - 1
        for i in range(length - 1):
            trans_table[item[i]][item[i + 1]] += 1
        trans_table[item[length - 1]][k] += 1
        for i in range(length):
            count[item[i]] += 1
        # for i in range(length - 1):
        #     count_clear[item[i]] += 1
    print("all_count =", all_count)
    print(trans_table)
    for i in range(k):
        tmp_sum = sum(trans_table[i])
        for j in range(k + 1):
            trans_table_string[i][j] = "{0}({1:.1f}%)".format(trans_table[i][j], 100 * trans_table[i][j] / tmp_sum)
    # print(np.asarray(trans_table))
    # order_list = [[trans_table[i][i], i] for i in range(k)]
    order_list = [[count[i], i] for i in range(k)]
    order_list.sort(key=lambda x: -x[0])
    order = [item[1] for item in order_list]
    trans_table_string_ordered = [[""] * (k + 1) for i in range(k)]
    for i in range(k):
        for j in range(k):
            trans_table_string_ordered[i][j] = trans_table_string[order[i]][order[j]]
        trans_table_string_ordered[i][k] = trans_table_string[order[i]][k]
    # print(np.asarray(trans_table_ordered))
    for i in range(k):
        for j in range(k + 1):
            print("{}\t".format(trans_table_string_ordered[i][j]), end="")
        print("{}".format(count[order[i]]), end="")
        print()


def one_time_draw_score():
    plt.figure(dpi=400, figsize=(8, 6))
    x = [3, 4, 5, 6, 7]
    y = [19/5, 38/8, 69/9, 78/7, 20/4]
    plt.plot(x, y, marker='s', markersize=5, linewidth=0.8, c="k", linestyle='dashed')
    xlabels = ["3", "4", "5", "6", "7"]
    # ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    # plt.figure()
    # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    # plt.imshow(data_normed, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.title("Average times variances in DPS subtypes less than those of K-means")
    plt.xticks(np.arange(3, 8, 1), xlabels)
    plt.ylim(0, 14)
    plt.xlabel("K")
    plt.ylabel("Times")
    for i in range(len(x)):
        plt.annotate("%.2f" % y[i], xy=(x[i], y[i]), xytext=(x[i] - 0.1, y[i] + 0.3))
    plt.show()


def one_time_draw_3_stairs(main_path):
    kmeans_label = np.load("test/kmeans_delta1_labels.npy", allow_pickle=True)
    flatten_label = np.load("test/sustain/delta1_final_16.npy")
    sustain_label = parse_flatten(flatten_label, "delta")
    # sustain_label = np.load("test/sustain_delta1_labels_2.npy", allow_pickle=True)
    dps_label = np.load("test/dps_delta1_labels_2.npy", allow_pickle=True)
    _, stairs_data_kmeans = get_heat_map_data_inter(main_path, 6, kmeans_label, "delta")
    _, stairs_data_sustain = get_heat_map_data_inter(main_path, 6, sustain_label, "delta")
    _, stairs_data_dps = get_heat_map_data_inter(main_path, 6, dps_label, "delta")
    draw_stairs_3(stairs_data_kmeans, stairs_data_sustain, stairs_data_dps, "test/final_inter/inter")
    return


def get_static_sustain(main_path, item_name):
    if item_name == "label":
        return np.load(main_path + "data/sustain/delta1_final_16_labels.npy", allow_pickle=True)
    if item_name == "intra":
        return np.load(main_path + "data/sustain/delta1_final_16_intra.npy", allow_pickle=True)
    if item_name == "inter":
        return np.load(main_path + "data/sustain/delta1_final_16_inter.npy", allow_pickle=True)


def coordinate(node1, node2, node3, node4):
    x1, y1 = node1[0], node1[1]
    x2, y2 = node2[0], node2[1]
    x3, y3 = node3[0], node3[1]
    x4, y4 = node4[0], node4[1]
    if x1 == x2 and x3 == x4:
        print("Error in coordinate (cover):", node1, node2, node3, node4)
        return [np.nan, np.nan]
    if y1 == y2 and y3 == y4:
        print("Error in coordinate (cover):", node1, node2, node3, node4)
        return [np.nan, np.nan]
    if x1 == x2:
        return [x1, (x4*y3-x3*y4-x1*y3+x1*y4)/(x4-x3)]
    if x3 == x4:
        return [x3, (x2*y1-x1*y2-x3*y1+x3*y2)/(x2-x1)]
    if y1 == y2:
        return [(y4*x3-y3*x4-y1*x3+y1*x4)/(y4-y3), y1]
    if y3 == y4:
        return [(y2*x1-y1*x2-y3*x1+y3*x2)/(y2-y1), y3]
    if (y1-y2)*(x4-x3) == (y3-y4)*(x2-x1):
        print("Error in coordinate (parallel):", node1, node2, node3, node4)
        return [np.nan, np.nan]
    else:
        a = ((x2*y1-x1*y2)*(x4-x3)-(x4*y3-x3*y4)*(x2-x1))/((y1-y2)*(x4-x3)-(y3-y4)*(x2-x1))
        b = (x2*y1-x1*y2-(y1-y2)*a)/(x2-x1)
        return [a, b]


def triangle_crd(nodes, one_label):
    rate01 = one_label[1] / (one_label[0] + one_label[1]) if one_label[0] + one_label[1] != 0 else 0.5
    # rate12 = one_label[2] / (one_label[1] + one_label[2]) if one_label[1] + one_label[2] != 0 else 0.5
    rate20 = one_label[0] / (one_label[2] + one_label[0]) if one_label[2] + one_label[0] != 0 else 0.5
    # print(rate01, rate12, rate20)
    m01 = [nodes[0][0] + rate01 * (nodes[1][0] - nodes[0][0]), nodes[0][1] + rate01 * (nodes[1][1] - nodes[0][1])]
    # m12 = [nodes[1][0] + rate12 * (nodes[2][0] - nodes[1][0]), nodes[1][1] + rate12 * (nodes[2][1] - nodes[1][1])]
    m20 = [nodes[2][0] + rate20 * (nodes[0][0] - nodes[2][0]), nodes[2][1] + rate20 * (nodes[0][1] - nodes[2][1])]
    # print(m01, m12, m20)
    res0 = coordinate(nodes[1], m20, nodes[2], m01)
    # res1 = coordinate(nodes[2], m01, nodes[0], m12)
    # res2 = coordinate(nodes[0], m12, nodes[1], m20)
    return res0


def get_triangle_data_y(main_path, label, data_name, k=6):
    data_y = np.load(main_path + "data/data_y/data_y_delta.npy".format(data_name[:-1]), allow_pickle=True)
    #print(data_y.shape)
    output = [[] for i in range(k)]
    z_raw = []
    for i, one_line in enumerate(label):
        for j, one_label in enumerate(one_line):
            # output[one_label].append(list(data_y[i][j]))
            z_raw.append(list(data_y[i][j]))
    data_all_trans = np.asarray(z_raw).swapaxes(0, 1)
    new_lines = []
    for line in data_all_trans:
        new_lines.append(np.abs((line - np.nanmean(line)) / np.nanstd(line)))
    data_all = np.asarray(new_lines).swapaxes(0, 1)
    tmp_index = 0
    for i, one_line in enumerate(label):
        for j, one_label in enumerate(one_line):
            output[one_label].append(list(data_all[tmp_index + j]))
        tmp_index += len(one_line)
    print([len(item) for item in output])
    return output


def get_triangle_data_x(main_path, label, data_name, k=6):
    data_x = np.load(main_path + "data/data_x/data_x_{}.npy".format(data_name), allow_pickle=True)
    # print(data_x.shape)
    output = [[] for i in range(k)]
    z_raw = []
    for i, one_line in enumerate(label):
        for j, one_label in enumerate(one_line):
            tmp_line = [np.mean(data_x[i][j][:148]), np.mean(data_x[i][j][148:296]), np.mean(data_x[i][j][296:])]
            z_raw.append(tmp_line)
            # tmp_line = np.asarray(tmp_line)
            # tmp_line = np.abs((tmp_line - np.mean(tmp_line)) / np.std(tmp_line))
            # tmp_line = list(tmp_line)
            #output[one_label].append(tmp_line)
    data_all_trans = np.asarray(z_raw).swapaxes(0, 1)
    new_lines = []
    for line in data_all_trans:
        new_lines.append(np.abs((line - np.nanmean(line)) / np.nanstd(line)))
    data_all = np.asarray(new_lines).swapaxes(0, 1)
    tmp_index = 0
    for i, one_line in enumerate(label):
        for j, one_label in enumerate(one_line):
            output[one_label].append(list(data_all[tmp_index + j]))
        tmp_index += len(one_line)
    # print([len(item) for item in output])
    return output


def draw_triangle(labels, label_type, save_path, show_flag=False):
    x_triangle = [0, 1, 2]
    y_triangle = [0, 1.732, 0]
    nodes = [[x_triangle[i], y_triangle[i]] for i in range(3)]
    color_types = ["red", "cyan", "blue", "green", "orange", "magenta", "yellow"]
    fig = plt.figure(dpi=400, figsize=(16, 9))
    # plt.title(title)
    ax_list = [231, 232, 233, 234, 235, 236]
    label_dic = {
        "data_x": ["A", "T", "N"],
        "data_y": ["CDRSB", "ADAS", "MMSE"]
    }
    offset_dic = {
        "data_x": [[-0.1, -0.1], [2, -0.1], [0.96, 1.8]],
        "data_y": [[-0.2, -0.1], [1.9, -0.1], [0.9, 1.8]]
    }
    for i, one_ax in enumerate(ax_list):
        ax = fig.add_subplot(one_ax)
        ax.plot(x_triangle, y_triangle, c="b", linewidth=2)
        ax.hlines(0, 0, 2, colors="b", linewidth=2)
        ax.set_title("Cluster #{} ({} Nodes)".format(i + 1, len(labels[i])), y=-0.1)
        for seat in ["left", "right", "top", "bottom"]:
            ax.spines[seat].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate(label_dic[label_type][0], xy=nodes[0], xytext=offset_dic[label_type][0])
        ax.annotate(label_dic[label_type][1], xy=nodes[1], xytext=offset_dic[label_type][1])
        ax.annotate(label_dic[label_type][2], xy=nodes[2], xytext=offset_dic[label_type][2])
        data = [triangle_crd(nodes, item) for item in labels[i]]
        data_x = [item[0] for item in data]
        data_y = [item[1] for item in data]
        ax.scatter(data_x, data_y, s=5, c=color_types[i])
    plt.savefig("{}".format(save_path), dpi=400)
    if show_flag:
        plt.show()


def build_table_data(main_path, label, data_name):
    pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    pt_dic = load_patient_dictionary(main_path, data_name[:-1])
    #
    # data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    # target_labels = CLINICAL_LABELS
    # data = data[["PTID", "EXAMDATE"] + target_labels + ["CDRSB", "ADAS13"]]
    # data = data[pd.notnull(data["EcogPtMem"])]
    #
    # for one_label in target_labels:
    #     data[one_label] = fill_nan(data[one_label])
    label_match = []
    for j, one_pt_id in enumerate(pt_ids):
        for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            label_match.append(label[j][k])
    data_x = load_data(main_path, "data/data_x/data_x_{}.npy".format(data_name))
    x_inter = []

    for j, one_pt_id in enumerate(pt_ids):
        for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
            # tmp = []
            # target_labels = target_labels[:7]
            # for one_target_label in target_labels:
            #     tmp.append(float(data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][
            #                          one_target_label].values[0]))
            x_inter.append(data_x[j][k])

    x_inter = np.array(x_inter)
    # print(np.shape(x_inter))
    # print(x_inter)
    # print(label_match)
    result = dict()
    result["sc"] = metrics.silhouette_score(x_inter, label_match, metric='euclidean')
    result["chi"] = metrics.calinski_harabasz_score(x_inter, label_match)
    result["dbi"] = metrics.davies_bouldin_score(x_inter, label_match)
    result["di"] = my_dunn_index(x_inter, label_match)
    for one_type in ["sc", "chi", "dbi", "di"]:
        print("%.6f\t" % result[one_type], end="")
    print()
    return result


def euclidean_dist(vec1, vec2):
    return np.linalg.norm(np.asarray(vec1) - np.asarray(vec2))


def my_dunn_index(x, label):
    inner_max = -1
    outer_min = 99999999
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if label[i] == label[j]:
                inner_max = max(inner_max, euclidean_dist(x[i], x[j]))
            else:
                outer_min = min(outer_min, euclidean_dist(x[i], x[j]))
                if euclidean_dist(x[i], x[j]) == 0:
                    print("label id = {} label = {} x = {}".format(i, label[i], x[i]))
                    print("label id = {} label = {} x = {}".format(j, label[j], x[j]))
                    print()
    return outer_min / inner_max


def entropy(pickle_file_path, output_folder_path):
    with open(pickle_file_path, "rb") as f:
        mydict = pickle.load(f)

    id, label, clinical_score = [], [], []
    for i, (k, v) in enumerate(mydict.items()):
        id.append(k)
        label.append(v["label"])
        clinical_score.append(v["clinical"])

    tmp = []
    df_id, df_label, df_label2, df_score = [], [], [], []
    for i in range(len(label) * 14):
        df_id.append(id[i // 14])
        df_label.append(label[i // 14])
        df_score.append(clinical_score[i // 14])
        if label[i // 14] == 1:
            tmp.append(clinical_score[i // 14][1])

    df = pd.DataFrame(df_score, columns=CLINICAL_LABELS)
    df["labels"] = df_label
    df["id"] = df_id

    df.to_csv(output_folder_path + "df.csv", index=False)
    tmp = pd.DataFrame(tmp)
    tmp.to_csv(output_folder_path + "tmp.csv", index=False)
    print(len(tmp), tmp)


def my_order(raw):
    raw = [[item, i] for i, item in enumerate(raw)]
    raw.sort(key=lambda x: x[0])
    raw = [[item[1], i] for i, item in enumerate(raw)]
    raw.sort(key=lambda x: x[0])
    return [item[1] for item in raw]


def one_time_score_step(matrix):
    tmp = 0
    for line in matrix:
        if len(set(line)) == 1:
            tmp += 0
        elif len(set(line)) == 2:
            tmp += 1
        else:
            tmp += 3
    return tmp


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    # pt_ids = np.load("data/ptid.npy", allow_pickle=True)
    # print(pt_ids)
    main_path = os.path.dirname(os.path.abspath("__file__")) + "/"
    # # dps_label = np.load("test/labels_40.npy", allow_pickle=True)
    # for i in [7,24,27,28,29,30,34,36,37,40,42,50]:
    #     dps_label = np.load("/Users/enze/Downloads/from termius/zeta2_k=6_50_epoch=3000/zeta2/{0}/labels_{0}.npy".format(i), allow_pickle=True)
    #     try:
    #         one_time_label_trans(dps_label, i)
    #     except Exception as e:
    #         print("{}\tSkipped:".format(i), e)
    # dps_label = np.load("/Users/enze/Downloads/from termius/zeta2_k=6_50_epoch=3000/zeta2/7/labels_7.npy",
    #                     allow_pickle=True)
    # l = []
    eta_paths = [
        "data=eta1_alpha=1e-05_beta=0.1_h_dim=8_main_epoch=1000",
        "data=eta1_alpha=1e-05_beta=0.01_h_dim=8_main_epoch=1000",
        "data=eta1_alpha=1e-05_beta=0.001_h_dim=8_main_epoch=1000",
        "data=eta2_alpha=1e-05_beta=1.0_h_dim=8_main_epoch=1000",
        "data=eta2_alpha=1e-05_beta=0.1_h_dim=8_main_epoch=1000",
        "data=eta2_alpha=1e-05_beta=0.01_h_dim=8_main_epoch=1000"
    ]
    # dps_label = np.load("/Users/enze/Downloads/from termius/{0}/{1}/{2}/labels_{2}.npy".format(eta_paths[1], eta_paths[1], 14),
    #                     allow_pickle=True)
    # _, heat_map_data_inter = get_heat_map_data_inter(main_path, 6, dps_label, "eta", True)
    # draw_stairs_3(heat_map_data_inter, heat_map_data_inter, heat_map_data_inter, "test/final_figure_inter", 0.05, True, True)


    #
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    # sustain_label = parse_flatten(flatten_label, "delta")
    # _, heat_map_data_inter_sustain = get_heat_map_data_inter(main_path, 6, sustain_label, "eta", True)
    # for path_id, one_path in enumerate(eta_paths):
    #     if path_id != 2:
    #         continue
    #     data_x_raw = load_data(main_path, "/data/data_x/data_x_{}_raw.npy".format(one_path[5:9]))
    #     kmeans_labels = get_kmeans_base(data_x_raw, 0, 6)
    #     _, heat_map_data_inter_kmeans = get_heat_map_data_inter(main_path, 6, kmeans_labels, "eta", True)
    #     for i in range(20):
    #         i += 1
    #         if i != 14:
    #             continue
    #         dps_label = np.load("/Users/enze/Downloads/from termius/{0}/{1}/{2}/labels_{2}.npy".format(one_path, one_path, i), allow_pickle=True)
    #         try:
    #             #one_time_label_trans(dps_label, "{0}/{1}/".format(one_path, i), 6, True)
    #             _, heat_map_data_inter_dps = get_heat_map_data_inter(main_path, 6, dps_label, "eta", False)
    #             print(heat_map_data_inter_dps)
    #             output_folder_path = "test/figure_inter_eta_normed_14/{0}/{1}/".format(one_path, i)
    #             if not os.path.exists(output_folder_path):
    #                 os.makedirs(output_folder_path)
    #             draw_stairs_3(heat_map_data_inter_kmeans, heat_map_data_inter_sustain, heat_map_data_inter_dps, output_folder_path + "inter",
    #                           0.05, False, False)
    #         except Exception as e:
    #             print("{}\t".format("{0}/{1}/".format(one_path, i)), end="")
    #             print(e)
    #     print("\n\n")


            # pkl_path = "/Users/enze/Downloads/from termius/{0}/{1}/{2}/dist/box_data_{3}_k=6_id={2}.pkl".format(
            #     one_path, one_path, i, one_path[5: 9])
            # output_folder_path = "test/entropy_eta/{0}/{1}/".format(one_path, i)
            # if not os.path.exists(output_folder_path):
            #     os.makedirs(output_folder_path)
            # try:
            #     print("id = ", i)
            #     # build_table_data(pkl_path, dps_label, "eta1")
            #     entropy(pkl_path, output_folder_path)
            # except Exception as e:
            #     print(e)

    # data_x_raw = load_data(main_path, "/data/data_x/data_x_{}_raw.npy".format("gamma4"))
    # for i in range(0, 5):
    #     print("seed =", i)
    #
    #     kmeans_labels = get_kmeans_base(data_x_raw, i, 6)
    #     _, heat_map_data_inter_kmeans = get_heat_map_data_inter(main_path, 6, kmeans_labels, "eta", True)
    #     draw_stairs_3(heat_map_data_inter_kmeans, heat_map_data_inter_kmeans, heat_map_data_inter_kmeans,
    #                   "test/trash/pic_2_{}".format(i),
    #                   0.05, True, False)

    # data_x_raw = load_data(main_path, "/data/data_x/data_x_zeta1_raw.npy")
    # kmeans_label = get_kmeans_base(data_x_raw, 0, 6)
    # np.save("test/kmeans_zeta1_labels.npy", kmeans_label, allow_pickle=True)
    # print(kmeans_label)
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    #
    # print([list(flatten_label).count(item) for item in range(6)])
    # # build_table_data(main_path, sustain_label, "delta1")
    # kmeans_label = np.load("test/kmeans_zeta1_labels.npy", allow_pickle=True)
    # kmeans_label_all = []
    # for item in kmeans_label:
    #     kmeans_label_all += list(item)
    #
    #
    # print([list(kmeans_label_all).count(item) for item in range(6)])
    # build_table_data(main_path, kmeans_label, "zeta1")
    # kmeans_label = np.load("test/kmeans_zeta1_labels.npy", allow_pickle=True)
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    # sustain_label = parse_flatten(main_path, flatten_label, "delta")

    for path_id, one_path in enumerate(eta_paths):
        for z in range(20):
            z += 1
            dps_label = np.load("/Users/enze/Downloads/from termius/{0}/{1}/{2}/labels_{2}.npy".format(one_path, one_path, z), allow_pickle=True)
            # dps_label = np.load("/Users/enze/Downloads/from termius/data=eta2_alpha=1e-05_beta=0.1_h_dim=8_main_epoch=1000/data=eta2_alpha=1e-05_beta=0.1_h_dim=8_main_epoch=1000/9/labels_9.npy", allow_pickle=True)
            # dps_label = np.load(
            #     "/Users/enze/Downloads/from termius/data=eta1_alpha=1e-05_beta=0.01_h_dim=8_main_epoch=1000/data=eta1_alpha=1e-05_beta=0.01_h_dim=8_main_epoch=1000/14/labels_14.npy",
            #     allow_pickle=True)
            data_type = "eta"
            label = dps_label
            K = 6
            pt_ids = np.load("data/ptid.npy", allow_pickle=True)
            pt_dic = load_patient_dictionary(main_path, data_type)

            data = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
            target_labels = CLINICAL_LABELS
            data = data[["PTID", "EXAMDATE"] + target_labels + ["MMSE", "CDRSB", "ADAS13"]]
            data = data[pd.notnull(data["EcogPtMem"])]

            for one_label in target_labels:
                data[one_label] = fill_nan(data[one_label])
            # print(data["MMSE"])
            # print(data["CDRSB"])
            # print(data["ADAS13"])
            result = []

            for i in range(K):
                dic = dict()
                for one_target_label in ["MMSE", "CDRSB", "ADAS13"]:
                    dic[one_target_label] = []
                for j, one_pt_id in enumerate(pt_ids):
                    for k, one_exam_date in enumerate(pt_dic.get(one_pt_id)):
                        if label[j][k] == i:
                            for one_target_label in ["MMSE", "CDRSB", "ADAS13"]:
                                tmp = data.loc[(data["PTID"] == one_pt_id) & (data["EXAMDATE"] == one_exam_date)][
                                    one_target_label].values[0]
                                dic[one_target_label] += [float(tmp)]

                #tmp_var = [np.var(np.asarray(dic[one_target_label])) for one_target_label in target_labels]
                tmp_avg = [np.nanmean(np.asarray(dic[one_target_label])) for one_target_label in ["MMSE", "CDRSB", "ADAS13"]]
                result.append(list(tmp_avg))
            try:
                result_col = []
                for i in range(3):
                    one_order = my_order([item[i] for item in result])
                    result_col.append(one_order)
                result_row = []
                for i in range(6):
                    result_row.append([item[i] for item in result_col])
                score = one_time_score_step(result_row)
                print(one_path, z, score)
                if score < 10:
                    for line in result_row:
                        for item in line:
                            print("%d\t" % item, end="")
                        print()
                    # print(result_row)
            except:
                pass
    # for i in range(3):
    #     one_order = my_order([item[i] for item in result])
    #     for item in one_order:
    #         print(item)
    #     print()

    # # build_table_data(main_path, dps_label, "delta1")
    #
    # # print(max(l))
    # make_heat_map_data_box(main_path, "test/test_box_zeta.pkl", dps_label, "eta")
    # draw_boxplt("test/test_box_zeta.pkl", "test/test_box/", "eta1", 6, 999)
    # build_data_x_y_iota(main_path)
    # data_x = load_data(main_path, "data/data_x/data_x_theta1.npy")
    # print(data_x.shape)
    # print(data_x)

    # build_data_x_y_eta(main_path)
    # build_data_x_y_zeta(main_path)
    # data_y = np.load(main_path + "data/data_y/data_y_{}.npy".format("zeta"), allow_pickle=True)
    # print(data_y.shape)
    # print(data_y)
    # triangle_crd([[0, 0], [2,0],[1,1.732]], [1,0.5,0.34])
    # dps_label = np.load("test/for_hist/labels_7.npy", allow_pickle=True)
    # kmeans_label = np.load("test/kmeans_delta1_labels.npy", allow_pickle=True)
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    # sustain_label = parse_flatten(flatten_label, "delta")
    # data_labels_x = one_time_build_triangle_data_x(sustain_label, "delta1", 6)
    # data_labels_y = one_time_build_triangle_data_y(sustain_label, "delta1", 6)
    # one_time_plot_triangle(data_labels_x, "data_x")
    # one_time_plot_triangle(data_labels_y, "data_y")
    # one_time_plot_triangle(1)
    # #dps_label = np.load("test/for_hist/labels_19.npy", allow_pickle=True)
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    # sustain_label = parse_flatten(flatten_label, "delta")
    # np.save("data/sustain/delta1_final_16_labels.npy", sustain_label, allow_pickle=True)
    # heat_map = get_heat_map_data(main_path, 6, sustain_label, "delta")
    # _, heat_map_inter = get_heat_map_data_inter(main_path, 6, sustain_label, "delta")
    # np.save("data/sustain/delta1_final_16_intra.npy", heat_map, allow_pickle=True)
    # np.save("data/sustain/delta1_final_16_inter.npy", heat_map_inter, allow_pickle=True)
    # one_time_label_trans(dps_label)
    # one_time_heat_map_data_box(main_path, "test/for_hist/box_data_delta1_k=6_3000_19.pkl", dps_label, "delta")
    # one_time_draw_3_stairs(main_path)
    # one_time_draw_score()
    # for i in range(11, 33):
    #     path = "test/sustain/delta1_final_{}.npy".format(i)
    #     flatten_label = np.load(path)
    #     print(path, [list(flatten_label).count(item) for item in range(6)])
    # 14, 16, 28, 30
    # flatten_label = np.load("test/sustain/delta1_final_16.npy")
    # sustain_label = parse_flatten(flatten_label, "delta")
    # # np.save("test/sustain_delta1_labels_2.npy", normal_label, allow_pickle=True)
    # # print(normal_label)
    # kmeans_label = np.load("test/kmeans_delta1_labels.npy", allow_pickle=True)
    # # sustain_label = np.load("test/sustain_delta1_labels_2.npy", allow_pickle=True)
    # # dtc_label = np.load("test/dtc_delta1_labels.npy", allow_pickle=True)
    # dps_label = np.load("test/dps_delta1_labels_2.npy", allow_pickle=True)
    # draw_320(dps_label, 6)
    # # one_time_label_trans_end_type(dps_label, 6)
    # # one_time_heat_map_data_box(main_path, dps_label, "delta")
    # # one_time_heat_map_data_box(main_path, dps_label, "delta")
    # heat_map_data_kmeans = get_heat_map_data(main_path, 6, kmeans_label, "delta")
    # heat_map_data_sustain = get_heat_map_data(main_path, 6, sustain_label, "delta")
    # # heat_map_data_dtc = get_heat_map_data(main_path, 6, dtc_label, "delta")
    # heat_map_data_dps = get_heat_map_data(main_path, 6, dps_label, "delta")
    # draw_heat_map_3(heat_map_data_kmeans, heat_map_data_sustain, heat_map_data_dps, "test/comparison", 2, True, 14)

    # draw_320(normal_label, 6)
    # data_x_raw = load_data(main_path, "/data/data_x/data_x_delta1_raw.npy")
    # kmeans_label = get_kmeans_base(data_x_raw, 0, 6)
    # np.save("test/kmeans_delta1_labels.npy", kmeans_label, allow_pickle=True)
    # draw_320(dps_label)
    # one_time_draw_3(main_path)
    # shutil.rmtree("test/todelete")
    # data = np.load("data/initial/alpha1/base_res.npy", allow_pickle=True)
    # draw_heat_map_2(data, data, "test/xx.png")
    # data = np.load("data/data_y/data_y_gamma.npy", allow_pickle=True)
    # print(data[0][0])
    # print(np.asarray([0,0,0,1959]).std())
    # build_data_x_y_gamma(main_path)
    # data_y = load_data(main_path, "/data/data_y/data_y_delta.npy")
    # print(data_y[46][0])
    # data_x_raw = load_data(main_path, "/data/data_x/data_x_eta2_raw.npy")
    # data = []
    # for item in data_x_raw:
    #     for line in item:
    #         data.append(line)
    # data = np.asarray(data)
    #
    # print(data.shape)
    # print(data[0])
    # np.save("test/sustain_eta2.npy", data, allow_pickle=True)

    # # label = np.load("test/labels_alpha2_k=5_2.npy", allow_pickle=True)
    # # # one_time_heat_map_data_box(main_path, 6, label, "delta")
    # #
    # one_time_tsne_data_x(main_path, label, "alpha2")
    # one_time_deal_tsne(50)
    # one_time_draw_tsne()
    # with open("test/data.dat", "r") as f:
    #     lines = f.readlines()
    # print(len(lines))
    # print([len(item) for item in lines])
    # print(lines)
    # data = np.load("test/labels.npy", allow_pickle=True)
    # draw_320(data)
    # with open("test/test_output_labels", "rb") as f:
    #     kmeans_labels = pickle.load(f)
    # s, res = get_heat_map_data_inter(main_path, 5, kmeans_labels, "delta")
    # for item in res:
    #     print(item)
    # print(res)
    # draw_stairs(res, res, "test/inter_cluster")
    # draw_stairs(1,1,1,1)
    # build_data_x_y_delta(main_path)
    # path = "saves/gamma1/1/proposed/trained/results/labels.npy"
    # data = np.load(path, allow_pickle=True)
    # print(data.shape)
    # print(data)
    # print(list(data))
    # print(data.shape)
    # print(split_periods_delta([20190401, 20190504, 20190607, 20190713]))
    # create_empty_folders_all(main_path)
    # draw_stairs()
    # count_pt_id_patient_lines(main_path)
    # build_data_x_alpha(main_path)
    # build_data_x_beta(main_path)
    # # build_patient_dictionary(main_path)
    # data_x = load_data(main_path, "/data/data_x/data_x_beta1.npy")
    # initial_record(main_path, data_x, "beta1", 0)
    # for item in CLINICAL_LABELS:
    #     print("{}_var,".format(item), end="")
    # build_data_y_beta(main_path)
    # data = np.load("data/cn_ad_labels_gamma.npy", allow_pickle=True)
    # print([len(item) for item in data])
    # print(data)
    # build_cn_ad_labels(main_path, "gamma")
    # data_x = load_data(main_path, "/data/data_x_new.npy")
    # base_res = np.load("data/initial/base_res.npy", allow_pickle=True)
    # #res = get_heat_map_data(main_path, 5, base_res)
    # draw_heat_map_2(base_res, base_res)
    # build_patient_dictionary(main_path)
    # pt_dic = load_patient_dictionary(main_path)
    # print(pt_dic)
    # print(type(pt_dic.get("002_S_0413")[0]))
    # print(pt_dic.keys())
    # enze_patient_data = np.load(main_path + "data/enze_patient_data_new.npy", allow_pickle=True)
    # pt_id_list = [item[0][3] for item in enze_patient_data]
    # # print(pt_id_list)
    # cn_ad_labels = get_cn_ad_labels(main_path, pt_id_list)
    # labels = np.load(main_path + 'saves/{}/proposed/trained/results/labels.npy'.format(1146))
    # print(create_label_string(labels, cn_ad_labels))
    # p = {
    #     "Cluster_std": 30,
    #     "MMSE_var": 50,
    #     "CDRSB_var": 20,
    #     "ADAS_var": 40
    # }
    # save_record(main_path, 10, 0, p, "test")
    # res1 = get_k_means_result(main_path)
    # res2 = get_ac_tpc_result(main_path, 1146)
    # draw_heat_map_2(res1, res2)
    # data = pd.read_excel("data/MRI_information_All_Measurement.xlsx", engine="openpyxl")
    # target_labels = ["MMSE", "CDRSB", "ADAS13"]
    # data = data[["PTID", "EXAMDATE"] + target_labels]
    # print(data)
    # print(data.dtypes)
    # data["PTID"] = data["PTID"].astype(str)
    # data["EXAMDATE"] = data["EXAMDATE"].astype(str)
    # print(data)
    # print(data.dtypes)
    # print(data.loc[(data["PTID"] == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])
    # print(data[(str(data["PTID"]) == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])

    # tmp = list(data.loc[(data["PTID"] == "002_S_0413")]["EXAMDATE"])
    # print("'{}'".format(tmp[-1]), type(tmp[-1]))
    pass










