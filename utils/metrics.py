import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import math
import random
import os
from torch_geometric.data import InMemoryDataset, Batch
def r_squared_error(y_true, y_pred):
    return r2_score(y_true, y_pred)

def get_k(y_true, y_pred):
    # 拟合 y_true = k*y_pred + b
    # 使用最小二乘法估计 k 和 b
    X = y_pred.reshape(-1, 1)
    y = y_true.reshape(-1, 1)

    # 计算 (X^T * X)^-1 * X^T * y
    # 如果 X^T * X 是奇异矩阵，使用伪逆
    try:
        k = np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError:
        k = np.linalg.pinv(X.T @ X) @ X.T @ y

    return k[0][0]


def get_cindex(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # Concordance index calculation
    n = len(y_true)
    if n < 2:
        return 0.0
    
    # All possible pairs (i, j) such that y_true[i] > y_true[j]
    # Here we are interested in pairs (i, j) such that y_true[i] != y_true[j]
    # and then check concordance based on y_pred
    
    c_index = 0.0
    consistent_pairs = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]: # Only consider informative pairs
                total_pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    consistent_pairs += 1
                elif y_pred[i] == y_pred[j]:
                    consistent_pairs += 0.5 # Handle ties in prediction
    
    if total_pairs == 0:
        return 0.0
    
    c_index = consistent_pairs / total_pairs
    return c_index


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def get_ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)



def model_evaluate(Y, P):

    return (get_mse(Y, P),
            get_rm2(Y, P),
            get_ci(Y, P))