import torch
import random
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

def compute_topological_centrality(mol, edge_index):
    """计算分子图的拓扑中心性 (degree + PageRank)"""
    num_nodes = mol.GetNumAtoms()
    edges = edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    deg_c = nx.degree_centrality(G)
    pr_c = nx.pagerank(G)
    deg_c = np.array([deg_c[i] for i in range(num_nodes)])
    pr_c = np.array([pr_c[i] for i in range(num_nodes)])

    # 归一化
    deg_c = (deg_c - deg_c.min()) / (deg_c.max() - deg_c.min() + 1e-9)
    pr_c = (pr_c - pr_c.min()) / (pr_c.max() - pr_c.min() + 1e-9)

    # 加权组合
    topo_c = 0.6 * deg_c + 0.4 * pr_c
    topo_c = np.power(topo_c, 1.2)
    return (topo_c - topo_c.min()) / (topo_c.max() - topo_c.min() + 1e-9)


def get_chemical_constraints(mol, u, v, lambda_qed=2.0):
    """
    基于 QED 子项的局部重要性 (自动计算结构权重)
    γ_uv = exp( λ * local_QED(u,v) )
    """
    try:
        # 计算 QED 的七个分量
        qed_components = QED.properties(mol)

        # 选取代表分子性质的特征项
        # logP: 脂溶性,  MW: 分子量, TPSA: 极性表面积
        logP = qed_components.ALOGP
        mw = qed_components.MW
        tpsa = qed_components.PSA
        arom = qed_components.AROM
        alerts = qed_components.ALERTS

        # 每个原子的局部重要性（估算）
        atom_importance = []
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            # 电负性近似: 原子序数归一化 + 极性因子
            polar_factor = 1.2 if z in [7, 8, 9, 15, 16] else 1.0
            aromatic_factor = 1.3 if atom.GetIsAromatic() else 1.0
            local_qed = (
                0.4 * logP +
                0.2 * (1 / (mw + 1e-6)) +
                0.2 * (1 / (tpsa + 1e-6)) +
                0.1 * arom -
                0.1 * alerts
            )
            atom_importance.append(local_qed * polar_factor * aromatic_factor)

        atom_importance = np.array(atom_importance)
        atom_importance = (atom_importance - atom_importance.min()) / (atom_importance.max() - atom_importance.min() + 1e-9)

        # 对边取平均并指数放缩
        local_qed_uv = (atom_importance[u] + atom_importance[v]) / 2.0
        gamma_uv = np.exp(lambda_qed * local_qed_uv)  # >1 表示应更保留
        return gamma_uv

    except Exception as e:
        return 1.0


def selective_dropout_adj(mol, edge_index, drop_prob=0.2, min_drop_ratio=0.05,
                          alpha=0.6, beta=(0.4, 0.2, 0.2, 0.2)):
    """
    CES引导的结构增强（按公式计算边保留概率）
    使用:
        p^e_uv = min( (S_max1 - log(w_uv)) / (S_max1 - u_s), p_r * p_e )
        w_uv = (φ_u + φ_v)/2
    其中 φ_u 为拓扑中心性
    """
    topo_c = compute_topological_centrality(mol, edge_index)
    topo_c = torch.tensor(topo_c, dtype=torch.float, device=edge_index.device)

    edges = edge_index.t().cpu().numpy()
    if len(edges) == 0:
        return edge_index

    S_max1 = 1.0
    u_s = 0.0
    p_r = 0.8
    p_e = drop_prob

    keep_edges = []

    bonds = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b for b in mol.GetBonds()}
    bonds.update({(b.GetEndAtomIdx(), b.GetBeginAtomIdx()): b for b in mol.GetBonds()})

    for (u, v) in edges:
        u, v = int(u), int(v)
        if (u, v) not in bonds:
            continue

        w_uv = (topo_c[u].item() + topo_c[v].item()) / 2.0

        # 使用 QED 自动权重
        chem_bonus = get_chemical_constraints(mol, u, v)

        w_uv_adjusted = w_uv * chem_bonus
        s_uv_adjusted = np.log(w_uv_adjusted)

        term1 = (S_max1 - s_uv_adjusted) / (S_max1 - u_s + 1e-9)
        term2 = p_r * p_e
        p_e_uv = min(term1, term2)

        keep_p = 1.0 - p_e_uv

        if random.random() < keep_p:
            keep_edges.append((u, v))

    min_edges = int(len(edges) * (1 - drop_prob - min_drop_ratio))
    if len(keep_edges) < min_edges:
        keep_edges = edges[:min_edges]

    new_edge_index = torch.tensor(keep_edges, dtype=torch.long).t().contiguous()
    return new_edge_index


def drop_feature(x, mol, edge_index, drop_prob=0.2, alpha=0.6, beta=(0.4, 0.2, 0.2, 0.2)):
    """
    CES引导的特征mask增强（仅使用拓扑中心性）
    """
    topo_c = compute_topological_centrality(mol, edge_index)
    topo_c = torch.tensor(topo_c, dtype=torch.float, device=x.device)

    node_drop_probs = drop_prob * (1.0 - topo_c)
    node_drop_probs = torch.clamp(node_drop_probs, 0.05, 0.9)

    x = x.clone()
    for i in range(x.size(0)):
        drop_mask = torch.rand((x.size(1),), device=x.device) < node_drop_probs[i]
        x[i, drop_mask] = 0.0
    return x
