import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_modules import GCNModel


class ContrastDrugModel(torch.nn.Module):
    #encoder_q是学生网络，encoder_k是教师网络，最后用于下游任务的是encoder_q
    def __init__(self, encoder_q,encoder_k, num_hidden, num_proj_hidden, tau=0.5,momentum=0.999):
        super(ContrastDrugModel, self).__init__()
        self.encoder_drug= encoder_q
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.momentum=momentum
        self.tau = tau

        # 投影头
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self,data_q,data_k):
        # 两个增强视图分别通过同一个 encoder_q
        node_q, graph_q = self.encoder_q(data_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            node_k, graph_k = self.encoder_k(data_k)
        drug_loss=self.compute_loss(node_q, graph_q, node_k, graph_k)
        return drug_loss

    # 投影到新的空间
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    # 余弦相似度
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # 半损失：正样本为 (z1[i], z2[i])，负样本为其他
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + 1e-8)
        )

    def loss(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        return ((l1 + l2) * 0.5).mean()

    # 图对比损失 (SimCLR)
    def graph_contrast_loss(self, z1, z2, tau=0.5, chunk_size=1024, neg_sample_size=1024, use_amp=True):
        """
        改进版图对比损失：使用 masked_fill 修复 IndexError 并支持下采样
        """
        z1 = F.normalize(self.projection(z1[-1]), dim=1)
        z2 = F.normalize(self.projection(z2[-1]), dim=1)
        N = z1.size(0)
        device = z1.device

        #正样本：同一图在两个视图中的相似度
        with torch.cuda.amp.autocast(enabled=use_amp):
            pos_sim = torch.sum(z1 * z2, dim=-1)  # [N]
            pos_exp = torch.exp(pos_sim / tau)  # [N]

        # 负样本 1: Cross-view (z1_i vs z2_j, j≠i)
        if neg_sample_size < N:
            neg_idx = torch.randperm(N, device=device)[:neg_sample_size]
            z2_neg = z2[neg_idx]

            cv_sim = torch.matmul(z1, z2_neg.T)
            cv_exp = torch.exp(cv_sim / tau)

            # 采样模式下的掩码：检查采样索引中是否包含当前节点索引
            mask = (torch.arange(N, device=device).view(-1, 1) == neg_idx.view(1, -1))
            cv_exp = cv_exp.masked_fill(mask, 0.0)  # 排除采样到的正样本
        else:
            cv_sim = torch.matmul(z1, z2.T)
            cv_exp = torch.exp(cv_sim / tau)

            # 全量模式下的掩码：排除对角线
            mask = torch.eye(N, device=device, dtype=torch.bool)
            cv_exp = cv_exp.masked_fill(mask, 0.0)

        cross_view_sum = cv_exp.sum(dim=1)  # [N]

        # 负样本 2: In-view (z1_i vs z1_k, k≠i)
        # 注意：为了显存安全，这里也可以考虑做采样，目前维持全量逻辑
        in_view_sim = torch.matmul(z1, z1.T)
        in_view_exp = torch.exp(in_view_sim / tau)

        # 排除对角线（自己 vs 自己）
        iv_mask = torch.eye(N, device=device, dtype=torch.bool)
        in_view_exp = in_view_exp.masked_fill(iv_mask, 0.0)
        in_view_sum = in_view_exp.sum(dim=1)  # [N]

        # InfoNCE 损失
        denom = cross_view_sum + in_view_sum + 1e-8
        loss = -torch.log(pos_exp / denom).mean()

        return loss

    # 节点对比损失（图内节点）
    def node_contrast_loss(self, node1, node2, tau=0.5, neg_sample_size=1024, use_amp=True):
        node1 = F.normalize(self.projection(node1[-1]), dim=1)
        node2 = F.normalize(self.projection(node2[-1]), dim=1)
        N = node1.size(0)
        device = node1.device

        with torch.cuda.amp.autocast(enabled=use_amp):
            # 1. 正样本 (View1_i vs View2_i)
            pos_sim = torch.sum(node1 * node2, dim=-1)
            pos_exp = torch.exp(pos_sim / tau)

            # 2. Cross-view 负样本采样逻辑
            if neg_sample_size < N:
                # 随机采样索引
                neg_idx = torch.randperm(N, device=device)[:neg_sample_size]
                node2_neg = node2[neg_idx]

                # 计算相似度 [N, neg_sample_size]
                cv_sim = torch.matmul(node1, node2_neg.T)

                # 动态生成 Mask：找到采样后的负样本中哪些是“自己”
                # node1 的索引是 0~N-1, node2_neg 的索引是 neg_idx
                mask = (torch.arange(N, device=device).view(-1, 1) == neg_idx.view(1, -1))

                cv_exp = torch.exp(cv_sim / tau)
                # 将正样本位置置零（不参与分母求和）
                cv_exp = cv_exp.masked_fill(mask, 0.0)
                cv_sum = cv_exp.sum(dim=1)
            else:
                # 全量模式逻辑
                cv_sim = torch.matmul(node1, node2.T)
                cv_exp = torch.exp(cv_sim / tau)
                mask = torch.eye(N, dtype=torch.bool, device=device)
                cv_exp = cv_exp.masked_fill(mask, 0.0)
                cv_sum = cv_exp.sum(dim=1)

            # 3. In-view 负样本
            iv_sim = torch.matmul(node1, node1.T)
            iv_exp = torch.exp(iv_sim / tau)
            iv_mask = torch.eye(N, dtype=torch.bool, device=device)
            iv_exp = iv_exp.masked_fill(iv_mask, 0.0)
            iv_sum = iv_exp.sum(dim=1)

            # 4. 计算 Loss
            denom = cv_sum + iv_sum + 1e-8
            loss = -torch.log(pos_exp / denom).mean()

        return loss

    def _momentum_update_key_encoder(self, init=False):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            if init:
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            else:
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def relational_graph_contrast_loss(self, graph_embs_q, graph_embs_k,
                                       temp_tk=0.04, temp_sq=0.1,
                                       chunk_size=512, use_amp=True):
        """
        关系对比损失（低显存 + AMP 优化版）

        参数：
            graph_embs_q: 强增强视图的图级特征（学生）
            graph_embs_k: 弱增强视图的图级特征（教师）
            temp_tk: 教师 softmax 的温度
            temp_sq: 学生 softmax 的温度
            chunk_size: 分块大小（控制显存）
            use_amp: 是否使用自动混合精度

        返回：
            loss: KL 散度对比损失
        """
        graph_embs_q = graph_embs_q[-1]
        graph_embs_k = graph_embs_k[-1]

        #投影 + 归一化
        z_q = F.normalize(self.projection(graph_embs_q), dim=1)
        z_k = F.normalize(self.projection(graph_embs_k), dim=1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # 学生（强增强）参与反向传播
            logits_q = torch.matmul(z_q, z_k.T)  # [B, B]
            # 教师（弱增强）不反传梯度
            with torch.no_grad():
                logits_k = torch.matmul(z_k, z_k.T)

            # 目标分布（教师）
            with torch.no_grad():
                P = F.softmax(logits_k / temp_tk, dim=1).detach()

            # 学生预测分布
            log_Q = F.log_softmax(logits_q / temp_sq, dim=1)

            # 分块计算 KL 散度
            loss_chunks = []
            B = P.size(0)
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                # KL散度: sum_i P_i * log(Q_i)
                loss_chunk = -torch.sum(P[start:end] * log_Q[start:end], dim=1)
                loss_chunks.append(loss_chunk)

            loss = torch.cat(loss_chunks, dim=0).mean()

        return loss

    # 总损失：节点对比 + 图对比 + MSE
    # 计算了每个损失项在总损失中的占比，然后加权求和
    def compute_loss(self, node_g1, graph_g1, node_g2, graph_g2,
                     ):
        loss_node = self.node_contrast_loss(node_g1, node_g2)
        loss_graph = self.graph_contrast_loss(graph_g1, graph_g2)
        loss_rel = self.relational_graph_contrast_loss(graph_g1, graph_g2, temp_tk=0.04,temp_sq=0.1)
        #print("loss_rel:", loss_rel.item())

        # with torch.no_grad():
        #     total = loss_node + loss_graph + loss_rel + 1e-8
        #     loss_node_norm = loss_node / total
        #     loss_graph_norm = loss_graph / total
        #     loss_rel_norm = loss_rel / total
        #
        # total_loss = (
        #           #loss_rel
        #         loss_node_norm * loss_node +
        #         loss_graph_norm * loss_graph
        #         +loss_rel_norm * loss_rel
        # )
        #total_loss=self.node_contrast_loss(node_g1, node_g2)+self.graph_contrast_loss(graph_g1, graph_g2)
        # 简单加权求和，避免梯度消失
        alpha_node = 0.3
        alpha_graph = 0.3
        alpha_rel = 0.4

        total_loss = (alpha_node * loss_node +
                      alpha_graph * loss_graph +
                      alpha_rel * loss_rel)

        return total_loss



class Contrast(nn.Module):
    """
    对比学习模块：
    接收 drug embedding + affinity embedding (za)
    接收 target embedding + affinity embedding (zb)
    使用 soft_pos 作为正样本权重进行 contrastive loss
    返回拼接的 fused embedding: [za_proj || zb_proj]
    """
    def __init__(self, hidden_dim, output_dim, tau=0.5, lam=0.5):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.tau = tau
        self.lam = lam

        # 初始化 Linear 层权重
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        计算 z1 和 z2 的 cosine 相似度矩阵并做温度缩放
        z1: [N, d]
        z2: [M, d]
        return: [N, M]
        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-8) / self.tau)
        return sim_matrix

    def forward(self, za, zd,zt, soft_pos,num_d):
        """
        za: [num_drug, d + d_aff]  (drug embedding + affinity embedding)
        zb: [num_target, d + d_aff]  (target embedding + affinity embedding)
        soft_pos: [num_drug, num_target]  正样本权重矩阵（可以是 S[i,j]）
        """
        # 投影
        zad=za[:num_d]
        zat=za[num_d:]
        zad_proj = self.proj(zad)  # [num_drug, output_dim]
        zat_proj = self.proj(zat)
        zd_proj = self.proj(zd)  # [num_target, output_dim]
        zt_proj = self.proj(zt)
        zad_mix_proj=torch.cat((zad_proj, zd_proj), 1)
        zat_mix_proj=torch.cat((zat_proj, zt_proj), 1)
        # 相似度矩阵
        sim_matrix = self.sim(zad_mix_proj, zat_mix_proj)  # [num_drug, num_target]

        # 构建 softmax 概率
        # sim_matrix_row = sim_matrix / (torch.sum(sim_matrix, dim=1, keepdim=True) + 1e-8)
        # sim_matrix_col = sim_matrix.t() / (torch.sum(sim_matrix.t(), dim=1, keepdim=True) + 1e-8)

        ###*******下面
        # P_row = F.softmax(sim_matrix / self.tau, dim=1)  # [num_drug, num_target]
        # P_col = F.softmax(sim_matrix.t() / self.tau, dim=1)  # [num_target, num_drug]
        # sim_matrix已经除以τ了，所以这边去掉
        # P_row = F.softmax(sim_matrix / self.tau, dim=1)  # [num_drug, num_target]
        # P_col = F.softmax(sim_matrix.t() / self.tau, dim=1)  # [num_target, num_drug]
        # 行方向归一化 (Drug -> Target)
        # dim=1 表示对每一行求和，keepdim=True 保持维度以便广播相除
        row_sums = torch.sum(sim_matrix, dim=1, keepdim=True)
        P_row = sim_matrix / (row_sums + 1e-8)

        # 列方向归一化 (Target -> Drug)
        # 先转置，再对行求和，再除
        sim_matrix_t = sim_matrix.t()
        col_sums = torch.sum(sim_matrix_t, dim=1, keepdim=True)
        P_col = sim_matrix_t / (col_sums + 1e-8)
        # *******上面

        # 使用 soft_pos 计算对比损失
        # loss_row = -torch.log((sim_matrix_row * soft_pos).sum(dim=1) + 1e-8).mean()
        # loss_col = -torch.log((sim_matrix_col * soft_pos.t()).sum(dim=1) + 1e-8).mean()
        loss_row = -(soft_pos * torch.log(P_row + 1e-8)).sum(dim=1).mean()
        loss_col = -(soft_pos.t() * torch.log(P_col + 1e-8)).sum(dim=1).mean()

        loss = self.lam * loss_row + (1 - self.lam) * loss_col

        # 拼接 embedding 返回
        # fused_embedding = torch.cat((za_proj, zb_proj), dim=1)  # [num_drug, output_dim*2]（或者 [num_drug+num_target, output_dim*2] 取决于下游使用方式）

        return loss, zad_mix_proj, zat_mix_proj
