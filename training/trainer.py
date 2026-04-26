import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch
from itertools import chain



class Trainer:
    def __init__(self, model, predictor, config, device):
        self.model = model
        self.predictor = predictor
        self.cfg = config
        self.device = device
        self.loss_fn = nn.MSELoss()

        # 仅优化需要梯度的参数
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())),
            lr=self.cfg['train']['lr'],
            weight_decay=0
        )

    def train_one_epoch(self, device,epoch, train_loader, drug_loaders, target_loaders, aff_graph, soft_pos):
        self.model.train()
        self.predictor.train()
        # 预处理图数据
        #drug_batches = [b.to(self.device) for b in drug_loaders]
        #target_batches = [b.to(self.device) for b in target_loaders]
        drug_batches = list(map(lambda graph: graph.to(device), drug_loaders))
        target_batches = list(map(lambda graph: graph.to(device), target_loaders))

        # 处理增强视图 g1, g2
        g1 = Batch.from_data_list(drug_batches)
        g1_list = [g1]

        g2 = Batch.from_data_list(drug_batches)
        g2_list = [g2]

        for batch_idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)

            ssl_loss, drug_emb, target_emb = self.model(
                aff_graph.to(self.device), drug_batches, target_batches, soft_pos, g1_list, g2_list
            )

            output, _ = self.predictor(data, drug_emb, target_emb)
            #loss = self.loss_fn(output, data.y.view(-1, 1).float()) + ssl_loss
            loss = self.loss_fn(output, data.y.view(-1, 1).float())
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def test(self, loader, drug_loaders, target_loaders, aff_graph, soft_pos):
        self.model.eval()
        self.predictor.eval()

        drug_batches = [b.to(self.device) for b in drug_loaders]
        target_batches = [b.to(self.device) for b in target_loaders]

        total_preds, total_labels = [], []
        with torch.no_grad():
            _, drug_emb, target_emb = self.model(
                aff_graph.to(self.device), drug_batches, target_batches, soft_pos, None, None
            )
            for data in loader:
                output, _ = self.predictor(data.to(self.device), drug_emb, target_emb)
                total_preds.append(output.cpu())
                total_labels.append(data.y.view(-1, 1).cpu())

        return torch.cat(total_labels).numpy().flatten(), torch.cat(total_preds).numpy().flatten()


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # 代表有改善
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False