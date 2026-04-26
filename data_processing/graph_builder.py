import os
import numpy as np
import networkx as nx
from collections import OrderedDict
from rdkit import Chem
from torch_geometric import data as DATA

# Assuming pro_res_table and other tables are imported from data_loader
from .data_loader import *

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

#分子图构建，使用 RDKit 从 SMILES 解析分子；
#每个原子的特征有 78 维（种类、度数、H数、价态、芳香性等 one-hot 编码）；
# 对每个原子提取特征（种类、度数、H数、价态、芳香性等 one-hot）；
# 构建邻接矩阵 mol_adj，并加上单位矩阵（自连接）；
def get_drug_molecule_graph(ligands):
    smile_graph = OrderedDict()
#使用 collections.OrderedDict 创建一个有序字典，保证插入顺序被保留
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
#将原始 SMILES 字符串解析为 RDKit 的分子对象（Mol）。验证 SMILES 是否有效，并构建分子拓扑结构。
        size, features, edge_index = smile_to_graph(lg)
        smile_graph[d] = (size, features, edge_index, lg)

#将标准化后的 SMILES 转换为图结构。
    return smile_graph

#蛋白质图构建
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
#构建边，包括自连接
    return c_size, features, edge_index

#蛋白质图构建
def get_target_molecule_graph(proteins, dataset):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
#MSA 用于提取进化信息（如共进化信号、序列保守性）。
#contac_path: 蛋白质接触图（Contact Map）预测结果的目录。
#接触图反映了蛋白质的3D 结构信息（即使没有实验结构也可预测）。
    target_graph = OrderedDict()
    for t in proteins.keys():
        g = target_to_graph(t, proteins[t], contac_path, msa_path)
        target_graph[t] = g

    return target_graph


def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
#构建接触图文件路径并加载数据
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
#矩阵中的值表示残基 i 和 j 是否在三维空间中接近
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
#最终 target_edge_index 是一个形状为 [num_edges, 2] 的 NumPy 数组。
    return target_size, target_feature, target_edge_index

#蛋白质节点特征提取：
def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
#根据目标蛋白的 ID（key）、序列、比对文件目录，自动找到对应的 .aln 文件，并调用上面的特征提取函数。
    feature = target_feature(aln_file, target_sequence)

    return feature

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
#计算 PSSM（Position-Specific Scoring Matrix，位置特异性打分矩阵）。
    # print(pssm.shape)
    other_feature = seq_feature(pro_seq)
    # print(other_feature.shape)
#输出：一个形状为 (L, 20) 的矩阵（L 是序列长度），表示每个位置对 20 种氨基酸的打分。
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)
