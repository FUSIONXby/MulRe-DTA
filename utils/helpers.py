import numpy as np
import torch
import math
import random
import os
from torch_geometric.data import InMemoryDataset, Batch
import numpy as np
from datetime import datetime
import pandas as pd

def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y

def denseAffinityRefine(adj, alpha=0.5):
    # 假设 adj 是邻接矩阵 (numpy array)
    # 将 adj 转换为 torch tensor
    adj_tensor = torch.from_numpy(adj).float()

    # D_hat = A + I
    adj_hat = adj_tensor + torch.eye(adj_tensor.size(0))

    # D_hat_inv_sqrt
    D_hat_diag = torch.sum(adj_hat, dim=1)
    D_hat_inv_sqrt = torch.diag(torch.pow(D_hat_diag, -0.5))

    # A_norm = D_hat_inv_sqrt * A_hat * D_hat_inv_sqrt
    adj_norm = torch.matmul(torch.matmul(D_hat_inv_sqrt, adj_hat), D_hat_inv_sqrt)

    return adj_norm.numpy()

def set_seed(seed=42):
    random.seed(seed)              # Python 内置随机模块
    np.random.seed(seed)           # Numpy 随机模块
    torch.manual_seed(seed)        # CPU 上的 torch
    torch.cuda.manual_seed(seed)   # 当前 GPU
    torch.cuda.manual_seed_all(seed)  # 所有 GPU（多卡情况）

    # 保证 cudnn 可复现（可能略微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量固定
    os.environ['PYTHONHASHSEED'] = str(seed)

def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def save_result_grouped_by_epoch(epoch_result, args, run_id, file_path="training_non.xlsx"):
    """
    保存训练结果，使相同 epoch 的不同 run 按行对齐：
    例如多次训练的 epoch=30 结果会堆在一起。
    """
    # 当前一条结果
    data = {
        "Epoch": epoch_result.get("epoch", None),
        "Run_ID": run_id,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "tau": getattr(args, "tau", None),
        "lam": getattr(args, "lam", None),
        "epochs": args.epochs,
    }
    data.update(epoch_result)

    new_df = pd.DataFrame([data])

    # 如果文件存在，加载并合并
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    # 对齐排序：先按 Epoch 排，再按 Run_ID 排
    #df = df.sort_values(by=["Epoch", "Run_ID"]).reset_index(drop=True)

    df.to_excel(file_path, index=False)
    print(f" 已保存结果到 {file_path} (Epoch {data['Epoch']}, Run {run_id})")