import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
import scipy.sparse as sp

#定义了一个“多层 GCN”的组合，作用在一个图上。
#对整张图聚合为一个图级表示。
class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()
        # 使用 nn.ModuleList 存储多个 GCNConv 层
        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            self.conv_layers.append(GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        """
        x: 节点特征 (num_nodes, in_dim)
        edge_index: 边索引 (2, num_edges)
        batch: batch vector (num_nodes,) 标记节点属于哪个图
        返回：
            graph_embeddings: list, 每层图级 embedding [(B, D_i), ...]
            node_embeddings: list, 每层节点 embedding [(num_nodes, D_i), ...]
        """
        output = x
        graph_embeddings = []
        node_embeddings = []

        for i, conv_layer in enumerate(self.conv_layers):
            output = conv_layer(output, edge_index, edge_weight)
            if i in self.relu_layers_index:
                output = self.relu(output)
            if i in self.dropout_layers_index:
                output = self.dropout(output)

            node_embeddings.append(output)           # 保存每层节点表示
            graph_embeddings.append(gep(output, batch))  # 聚合成图级表示

        return graph_embeddings, node_embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        """
        graph_batchs: list of Data 对象 (每个图)
        返回：
            embeddings_per_layer: list, 每层图级 embedding [(B, D_i), ...]
            node_embeddings_per_layer: list, 每层节点 embedding [(total_nodes, D_i), ...]
        """
        all_graph_embeddings = []
        all_node_embeddings = []

        # 遍历每个图
        for graph in graph_batchs:
            g_emb, n_emb = self.graph_conv(graph.x, graph.edge_index, None, graph.batch)
            all_graph_embeddings.append(g_emb)  # 每个元素是 list: 每层的图级 embedding
            all_node_embeddings.append(n_emb)   # 每个元素是 list: 每层节点 embedding

        # 对每层拼接 batch 中所有图
        embeddings_per_layer = []
        node_embeddings_per_layer = []
        for layer_idx in range(self.num_layers):
            # 图级 embedding
            layer_graph_embs = [g_emb[layer_idx] for g_emb in all_graph_embeddings]
            embeddings_per_layer.append(torch.cat(layer_graph_embs, dim=0))

            # 节点级 embedding
            layer_node_embs = [n_emb[layer_idx] for n_emb in all_node_embeddings]
            node_embeddings_per_layer.append(torch.cat(layer_node_embs, dim=0))

        return embeddings_per_layer, node_embeddings_per_layer

#与 GCNBlock 类似，但用的是 DenseGCNConv，支持稠密邻接矩阵。
class DenseGCNBlock(nn.Module):
#密集（Dense）版本的 GCN 层（即 DenseGCNConv），适用于邻接矩阵形式的图数据
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
#x为节点特征矩阵，adj为邻接矩阵，DenseGCNConv 要求输入是 3D 张量
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings

#使用 dropout_adj 对亲和力图的边做 Dropout；输入图结构包含药物+蛋白质联合图（adj, x）。
class DenseGCNModel(nn.Module):
#用于建模药物（drug）和靶标（target）之间的相互作用图
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
#这个adj就是亲和力图的邻接矩阵，adj: 邻接矩阵，维度 [num_d + num_t, num_d + num_t]
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
#上面这四个变量分别代表：节点特征矩阵、邻接矩阵、药物节点数、蛋白质节点数
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
#上面两行提取非零边（原始连接）
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
#对边进行Dropout,输出：edge_indexs_dropout → 丢弃后剩下的边。edge_weights_dropout → 对应的边权（原亲和力值）。
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout
#对边进行 Dropout，随机丢弃 p 比例的边。
        embeddings = self.graph_conv(xs, adj_dropout)
#得到一个“稀疏化”的邻接矩阵，用于图卷积。
        return embeddings


