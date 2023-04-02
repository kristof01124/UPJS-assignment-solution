# Kristof code
import copy
from heapq import merge

import numpy as np

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re

from sklearn.svm import OneClassSVM


def minibatch_kms_clustering(input: DataFrame, scaled: list):
    data = copy.deepcopy(input)
    kmeans = MiniBatchKMeans(n_clusters=5, batch_size=1000, max_iter=100)
    prediction = kmeans.fit_predict(scaled)
    data['outlier'] = prediction
    return data;


def lof_clustering(input: DataFrame, scaled: list):
    data = copy.deepcopy(input)
    lof = LocalOutlierFactor(n_neighbors=20)
    prediction = lof.fit_predict(scaled)
    lof.fit(scaled)
    data['outlier'] = prediction
    return data


def iforest_clustering(input: DataFrame, scaled: list):
    data = copy.deepcopy(input)
    n_estimators = 100
    max_samples = 256
    iForest = IsolationForest(contamination=0.006, n_estimators=n_estimators, max_samples=max_samples)
    iForest.fit(scaled)
    prediction = iForest.predict(scaled)
    data['outlier'] = prediction
    return data


def one_class_SVM_clustering(input: DataFrame, scaled: list):
    data = copy.deepcopy(input)
    nu = 0.5
    kernel = 'linear'
    gamma = 'auto'
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(scaled)
    prediction = ocsvm.predict(scaled)
    data['outlier'] = prediction
    return data


def read_data():
    file = pd.read_csv('./data/dc_file_modified2.csv')
    array = []
    for i in range(len(file['filename'])):
        array.append(str(file['inode'][i]) + '-' + file['filename'][i])  # alt + 4571
    file['filenames-inode'] = array
    parameters = ['M',
                  'A',
                  'C',
                  'B',
                  'file_stat',
                  'NTFS_file_stat',
                  'file_entry_shell_item',
                  'NTFS_USN_change',
                  'dir_appdata',
                  'dir_win',
                  'dir_user',
                  'dir_other',
                  'file_executable',
                  'file_graphic',
                  'file_documents',
                  'file_ps',
                  'file_other',
                  'mft',
                  'lnk_shell_items',
                  'olecf_olecf_automatic_destinations/lnk/shell_items',
                  'winreg_bagmru/shell_items',
                  'usnjrnl',
                  'is_allocated1',
                  'is_allocated0']
    file_grouped = file.groupby(['filenames-inode'], as_index=False).sum(parameters)
    sub_file = file_grouped[parameters]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(file_grouped[parameters])
    return (file, file_grouped, scaled)


def get_outliers(data: DataFrame):
    return data[data['outlier'] == -1]['filenames-inode']


def get_FCA_outliers(file: DataFrame):
    sub_file = file[[
        'inode',
        'M',
        'A',
        'C',
        'B',
        'file_stat',
        'NTFS_file_stat',
        'file_entry_shell_item',
        'NTFS_USN_change', 'filef',
        'directory',
        'link',
        'dir_appdata',
        'dir_win',
        'dir_user',
        'dir_other',
        'file_executable',
        'file_graphic',
        'file_documents',
        'file_ps',
        'file_other',
        'mft',
        'lnk_shell_items',
        'olecf_olecf_automatic_destinations/lnk/shell_items',
        'winreg_bagmru/shell_items',
        'usnjrnl',
        'is_allocated1',
        'is_allocated0',
        'file_size',
        'filename'
    ]]

    sub_file["filenames-inode"] = sub_file['inode'].astype(str) + "-" + sub_file["filename"]
    sub_file = sub_file.drop(['inode'], axis=1)
    sub_file = sub_file.drop(['filename'], axis=1)

    sizes = np.array(sub_file[sub_file.file_size > 0].file_size)
    q1 = np.percentile(sizes, 25)
    q2 = np.percentile(sizes, 50)
    q3 = np.percentile(sizes, 75)
    sub_file["size_none"] = [1 if s else 0 for s in (sub_file.file_size == 0)]
    sub_file["size_Q1"] = [1 if s else 0 for s in (0 < sub_file.file_size) & (sub_file.file_size < q1)]
    sub_file["size_Q2"] = [1 if s else 0 for s in (q1 <= sub_file.file_size) & (sub_file.file_size < q2)]
    sub_file["size_Q3"] = [1 if s else 0 for s in (q2 <= sub_file.file_size) & (sub_file.file_size < q3)]
    sub_file["size_Q4"] = [1 if s else 0 for s in q3 <= sub_file.file_size]

    sub_file.rename(columns={'olecf_olecf_automatic_destinations/lnk/shell_items': "olect_shell_items",
                             "winreg_bagmru/shell_items": "winreg_bagmru"})

    sub_file = sub_file.drop_duplicates()

    sub_file["Assoc_rule_1"] = [0 if (c + b == 2 and a == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_2"] = [0 if (a + c + b == 3 and m == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_3"] = [0 if (m + b == 2 and a == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_4"] = [0 if (a + c == 2 and m == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_5"] = [0 if (m + a + c == 3 and b == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_6"] = [0 if (m + c == 2 and a == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]
    sub_file["Assoc_rule_7"] = [0 if (m + a + c == 3 and b == 0) else 1 for m, a, c, b in
                                zip(sub_file.M, sub_file.A, sub_file.C, sub_file.B)]

    sub_file["Assoc_rule_8"] = [0 if (c + b + sn == 3 and a == 0) else 1 for c, b, sn, a in
                                zip(sub_file.C, sub_file.B, sub_file.size_none, sub_file.A)]
    sub_file["Assoc_rule_9"] = [0 if (d == 1 and f == 0) else 1 for d, f in
                                zip(sub_file.dir_win, sub_file.file_stat)]
    sub_file["Assoc_rule_10"] = [0 if (a + f + d + dir + sn == 5 and m == 0) else 1 for a, f, d, dir, sn, m in
                                 zip(sub_file.A, sub_file.file_stat, sub_file.directory, sub_file.dir_win,
                                     sub_file.size_none, sub_file.M)]
    sub_file["Assoc_rule_11"] = [0 if (a + d + s == 3 and n + m < 2) else 1 for a, d, s, n, m in
                                 zip(sub_file.A, sub_file.dir_other, sub_file.size_none, sub_file.NTFS_file_stat,
                                     sub_file.mft)]
    sub_file["Assoc_rule_12"] = [0 if (c + b + d == 3 and m + a < 2) else 1 for c, b, d, m, a in
                                 zip(sub_file.C, sub_file.B, sub_file.dir_user, sub_file.M, sub_file.A)]
    sub_file["Assoc_rule_13"] = [0 if (f + ff + fff == 3 and d == 0) else 1 for f, ff, fff, d in
                                 zip(sub_file.file_stat, sub_file.filef, sub_file.file_executable, sub_file.dir_win)]
    sub_file["Assoc_rule_14"] = [0 if (c + f + ff + fff == 4 and d == 0) else 1 for c, f, ff, fff, d in
                                 zip(sub_file.C, sub_file.file_stat, sub_file.filef, sub_file.file_executable,
                                     sub_file.dir_win)]
    sub_file["Assoc_rule_15"] = [0 if (a + f + ff + fff == 4 and d == 0) else 1 for a, f, ff, fff, d in
                                 zip(sub_file.A, sub_file.file_stat, sub_file.filef, sub_file.file_executable,
                                     sub_file.dir_win)]
    sub_file["Assoc_rule_16"] = [0 if (q + f + ff + fff == 4 and d == 0) else 1 for q, f, ff, fff, d in
                                 zip(sub_file.size_Q4, sub_file.file_stat, sub_file.filef, sub_file.file_executable,
                                     sub_file.dir_win)]

    data_w_assoc = sub_file[["filenames-inode"] + ["Assoc_rule_" + str(i) for i in range(1, 17)]]
    data = data_w_assoc.drop(["filenames-inode"], axis=1)

    # plt.matshow(data.corr())
    # plt.show()

    return sub_file[data.sum(axis=1) < 14]["filenames-inode"]


def find_matching_outliers(dataset1: list, dataset2: list):
    print("OUTLIERS:")
    counter = 0
    for outlier in dataset1:
        if outlier in dataset2:
            counter += 1
            print(f"MATCH! no. {counter} : {outlier}")


def analyze_cluster(dataset: DataFrame):
    print('Number of outliers: {}'.format(len(dataset[dataset['outlier'] == -1])))


def insert_data_to_output(title: str, cluster_data: DataFrame, data: list):
    col = [0 for i in range(len(data))]
    try:
        outliers = get_outliers(cluster_data)
    except:
        outliers = cluster_data

    print('Started inserting')
    index = 0
    for i in outliers:
        col[data.index(i)] = 1
    print('Done labeling')
    output[title] = col
    print('Done inserting')


file, data, scaled = read_data()
print(scaled)
FCA_outliers = list(get_FCA_outliers(file))
kms_cluster_data = minibatch_kms_clustering(data, scaled)

print("Iforest clustering")
iforest_cluster_data = iforest_clustering(data, scaled)
analyze_cluster(iforest_cluster_data)
find_matching_outliers(get_outliers(iforest_cluster_data), FCA_outliers)

print("Lof clustering")
lof_cluster_data = lof_clustering(data, scaled)
analyze_cluster(lof_cluster_data)
find_matching_outliers(get_outliers(lof_cluster_data), FCA_outliers)

outliers = set()
outliers = outliers.union(get_outliers(lof_cluster_data))
outliers = outliers.union(get_outliers(iforest_cluster_data))
outliers = outliers.union(FCA_outliers)

outliers = list(outliers)

output = DataFrame()

output['id'] = outliers
insert_data_to_output('Iforest', iforest_cluster_data, outliers)
insert_data_to_output('LOF', lof_cluster_data, outliers)
insert_data_to_output('Matje', FCA_outliers, outliers)

output.to_csv('output.csv')
