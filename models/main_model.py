import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_modules import GCNModel, DenseGCNModel
from .prediction_head import PredictModule
from .contrastive_learning import *
class DTA_GCN(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.2):
        super(DTA_GCN, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        #self.drug_graph_conv = GCNModel(d_ms_dims)
        self.target_graph_conv = GCNModel(t_ms_dims)
        #下面是两个对比学习模块，ns_dims是亲和力图（affinity graph）的 GCN 层维度
        #药物对比
        #靶标对比
        # self.drug_contrast = Contrast2(ns_dims[-1], embedding_dim, tau, lam)
        # self.target_contrast = Contrast2(ns_dims[-1], embedding_dim, tau, lam)
        self.cl_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)

        self.encoder_q = GCNModel(d_ms_dims)
        self.encoder_k = GCNModel(d_ms_dims)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.drug_node_graph_contrast=ContrastDrugModel(encoder_drug=self.drug_graph_conv,num_hidden=256,num_proj_hidden=64,tau=0.5)
        # self.drug_node_graph_contrast = ContrastDrugModel(encoder_q=self.encoder_q, encoder_k=self.encoder_k,
        #                                                   num_hidden=256, num_proj_hidden=64, tau=0.5, momentum=0.999)
        self.drug_node_graph_contrast = ContrastDrugModel(encoder_q=self.encoder_q, encoder_k=self.encoder_k,
                                                          num_hidden=256, num_proj_hidden=64, tau=0.5, momentum=0.999)

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, soft_pos,g1,g2):
        num_d = affinity_graph.num_drug
        #print("affinity_graph", affinity_graph)#510*510
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]#510,256
        #drug_graph_embedding,_ = self.drug_graph_conv(drug_graph_batchs)[-1]
        #Dgraph_embeddings, Dnode_embeddings = self.drug_graph_conv(drug_graph_batchs)
        Dgraph_embeddings, Dnode_embeddings = self.encoder_q(drug_graph_batchs)
        drug_graph_embedding = Dgraph_embeddings[-1]  # shape (B, D_last)
        #target_graph_embedding,_ = self.target_graph_conv(target_graph_batchs)[-1]
        Tgraph_embeddings, Tnode_embeddings = self.target_graph_conv(target_graph_batchs)
        target_graph_embedding = Tgraph_embeddings[-1]  # shape (B, D_last)
        #print("affinity_graph_embedding[:num_d]的维度", affinity_graph_embedding[:num_d].shape)
        #print("pos的维度", drug_pos.shape)#68,68
        #print("pos的维度", target_pos.shape)#442,442
#2. 药物对比损失 + 嵌入
        #drug_embedding:68,256
        #drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding)
#3. 蛋白质对比损失 + 嵌入
        #tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding)
        cl_loss, drug_embedding, target_embedding = self.cl_contrast(affinity_graph_embedding, drug_graph_embedding,
                                                                   target_graph_embedding, soft_pos, num_d)
        if self.training:  # 只在训练时执行
            #drug_loss = self.drug_node_graph_contrast(g1, g2)
            cl_loss, drug_embedding, target_embedding = self.cl_contrast(affinity_graph_embedding, drug_graph_embedding,
                                                                  target_graph_embedding, soft_pos, num_d)
            #cl_loss=drug_loss+cl_loss
        else: cl_loss=0

        #return dru_loss + tar_loss, drug_embedding, target_embedding
        #return cl_loss, drug_embedding, target_embedding
        return 0,drug_embedding, target_embedding

