import pickle
import json
import numpy as np
import torch
from torch_geometric import data as DATA
from collections import OrderedDict
from Aload2.utils.helpers import *
from Aload2.utils.datasets import *
#字典归一化，用于对各种氨基酸性质表（如分子量、pKa 等）进行归一化处理
#归一化缩放到0-1
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0

    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
#####下面氨基酸分类表
#脂肪族残基，非芳香
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
#芳香族残基
pro_res_aromatic_table = ['F', 'W', 'Y']
#极性中性残基
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
#酸性带电残基
pro_res_acidic_charged_table = ['D', 'E']
#碱性带电残基
pro_res_basic_charged_table = ['H', 'K', 'R']
####上面氨基酸分类表

###理化性质查找表
#氨基酸的分子量、pKa、pKb、pKx、等电点（pl）、亲水性（pH=2 和 pH=7）等
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
###理化性质查找表


res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

#加载亲和力数据：load_data()。加载 .pkl 格式的亲和力矩阵（如 Davis、KIBA）。
#若是 Davis 数据集，对原始 Kd 取 -log10(Kd/1e9) 转换成 pKd。
def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity

#数据划分与处理：process_data()，划分训练集和测试集（根据 json 文件的索引）；
# 构造 DTADataset，存储配对样本（drug_id、target_id、affinity）；
# 构建亲和力图 affinity_graph，以及药物和蛋白质的 k-NN 相似图 drug_pos, target_pos
def process_data(affinity_mat, dataset, pos_threshold):
    dataset_path = 'data/' + dataset + '/'

    # train_file = json.load(open(dataset_path + 'train_set.txt'))
    train_index = json.load(open(dataset_path + 'train_set.txt'))
    # train_index = [idx for sublist in train_file for idx in sublist]
    test_index = json.load(open(dataset_path + 'test_set.txt'))
    val_index = json.load(open(dataset_path + 'val_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)
    val_rows, val_cols = rows[val_index], cols[val_index]
    val_Y = affinity_mat[val_rows, val_cols]
    val_dataset = DTADataset(drug_ids=val_rows, target_ids=val_cols, y=val_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph, soft_pos= get_affinity_graph_with_soft_labels(dataset, train_affinity_mat, pos_threshold,alpha=0.5)

    return train_dataset, val_dataset, test_dataset, affinity_graph,soft_pos

#亲和力图构建，用于构建一个异构图：
# 节点 = 药物节点 + 蛋白质节点；
# 边 = 药物-蛋白质之间的亲和力值；
# 额外还构建了药物-药物和蛋白质-蛋白质的 k-NN 相似图。

def get_affinity_graph_with_soft_labels(dataset, adj, pos_threshold, alpha=0.5):
    """
    dataset: 数据集名
    adj: [num_drug, num_target] 原始亲和力矩阵
    num_pos: 每个节点最多保留的正连接数量（稀疏化）
    pos_threshold: 判断强结合阈值
    alpha: 融合药物相似性和蛋白质相似性的权重
    """

    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    # # --- Step 1: 二值化 adj，得到强结合 mask ---
    dt_bin = np.where(adj >= pos_threshold, 1.0, 0.0)  # [num_drug, num_target]
    #
    # # --- Step 2: 构建药物-药物相似性 dtAll ---
    dtd = np.matmul(dt_bin, dt_bin.T)  # 共靶标数
    dtd = dtd / dtd.sum(axis=-1, keepdims=True)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug)  # 自身连接
    dtd = dtd.astype("float32")
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')  # 外部相似性
    dAll = dtd + d_d  # 内部协同 + 外部相似性
    #
    # # --- Step 3: 构建蛋白质-蛋白质相似性 tAll ---
    td_bin = dt_bin.T  # [num_target, num_drug]
    tdt = np.matmul(td_bin, td_bin.T)
    tdt = tdt / tdt.sum(axis=-1, keepdims=True)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target)
    tdt = tdt.astype("float32")
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = tdt + t_t
    #dAll=np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')  # 外部相似性
    #tAll=np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')

    # --- Step 4: 构造 pair-level soft label S ---
    # dt_bin: [num_drug, num_target] 强结合 mask
    # dAll: [num_drug, num_drug] 药物相似
    # tAll: [num_target, num_target] 蛋白质相似
    # 融合传播：
    # S[i,j] = α * Σ_{d'} dAll[i,d'] * dt_bin[d',j] + (1-α) * Σ_{t'} dt_bin[i,t'] * tAll[t',j]
    S_drug = np.matmul(dAll, dt_bin)      # 药物相似传播
    S_target = np.matmul(dt_bin, tAll)    # 蛋白相似传播
    S = alpha * S_drug + (1 - alpha) * S_target  # [num_drug, num_target]

    # --- Step 5: 归一化 S → soft label ---
    S = S / (S.sum(axis=1, keepdims=True) + 1e-8)  # 每行归一化
    S = np.nan_to_num(S)
    #
    # S = sp.coo_matrix(S)
    # S=sparse_mx_to_torch_sparse_tensor(S)
    # 稀疏化：每个药物只保留 top-num_pos 邻居
    # drug_pos = sp.coo_matrix(drug_pos)
    # drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

    # --- Step 6: 构建 bipartite affinity graph（原有逻辑保持） ---
    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)

    adj_1 = adj_norm
    adj_2 = adj_norm.T
    adj_full = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)

    train_row_ids, train_col_ids = np.where(adj_full != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj_full[train_row_ids, train_col_ids]

    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj_full)
    adj_features[adj_full != 0] = 1
    features = np.concatenate((node_type_features, adj_features), 1)

    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj_full),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, S  # 返回 bipartite 图 + pair-level soft label

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]

    return np.array(res_property1 + res_property2)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))



def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        lines = f.readlines() # Read all lines once
        line_count = len(lines)
        for line in lines:
            line = line.strip()
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat
