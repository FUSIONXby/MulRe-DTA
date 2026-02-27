import pandas as pd
import numpy as np
import re
from rdkit import Chem
from collections import Counter
import torch
import torch.nn as nn

# -----------------------------
# 1. 数据加载和预处理函数
# -----------------------------

def tokenize_smiles(smiles):
    # 支持识别多字符原子（如 Cl、Br）、括号、数字等
    pattern = r"($.*?$|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|$|$|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>|\*|\$|_|\{|\}|\%)"
    tokens = [token for token in re.findall(pattern, smiles) if token]
    return tokens

# 假设你有如下变量定义
file_path = "data/Davis/DTA"
dataset = "davis"  # 这个值根据你的实际数据集名称调整

# 加载训练集和测试集
train_data = pd.read_csv(f"{file_path}/{dataset}_train.csv")
test_data = pd.read_csv(f"{file_path}/{dataset}_test.csv")

# 获取 SMILES 列（列名固定为 COMPOUND_SMILES）
train_smiles_list = train_data['COMPOUND_SMILES'].tolist()
test_smiles_list = test_data['COMPOUND_SMILES'].tolist()

# 构建词汇表：只在训练集上构建
all_tokens = []
for smi in train_smiles_list:
    tokens = tokenize_smiles(smi)
    all_tokens.extend(tokens)

vocab_counter = Counter(all_tokens)
vocab = ['<PAD>', '<UNK>'] + list(vocab_counter.keys())
token2idx = {token: idx for idx, token in enumerate(vocab)}
vocab_size = len(token2idx)

# 序列最大长度（基于训练集确定）
max_len = max(len(tokenize_smiles(smi)) for smi in train_smiles_list)

# 编码函数
def encode_smiles(smiles, token2idx, max_len):
    tokens = tokenize_smiles(smiles)
    encoded = [token2idx.get(t, token2idx['<UNK>']) for t in tokens]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded += [token2idx['<PAD>']] * (max_len - len(encoded))
    return encoded

# 对训练集和测试集分别编码
X_train = np.array([encode_smiles(smi, token2idx, max_len) for smi in train_smiles_list])
X_test = np.array([encode_smiles(smi, token2idx, max_len) for smi in test_smiles_list])

print("Train Encoded Shape:", X_train.shape)
print("Test Encoded Shape:", X_test.shape)

# -----------------------------
# 2. 定义 LSTM 模型
# -----------------------------
class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=80, num_layers=2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token2idx['<PAD>'])
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x):
        x = self.embedding(x)  # [B, T, E]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, H*2], h_n: [L*2, B, H]
        # 取最后时刻的隐藏状态拼接作为整个序列的表示
        batch_size = x.size(0)
        return lstm_out, h_n.transpose(0, 1).reshape(batch_size, -1)

# -----------------------------
# 3. 构建数据集和模型
# -----------------------------
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

train_dataset = SmilesDataset(X_train)
test_dataset = SmilesDataset(X_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LSTMEmbedding(vocab_size=vocab_size, embed_dim=64, hidden_dim=80, bidirectional=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 4. 推理阶段：提取特征
# -----------------------------
def extract_features(loader):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, final_hidden = model(batch)
            all_embeddings.append(final_hidden.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    return embeddings

# 提取训练集和测试集的嵌入
train_embeddings = extract_features(train_loader)
test_embeddings = extract_features(test_loader)

print("Train Embeddings Shape:", train_embeddings.shape)
print("Test Embeddings Shape:", test_embeddings.shape)

save_path = r"data\Davis\processed\test\compound_embedding.npy"
np.save(save_path,test_embeddings)

save_path2 = r"data\Davis\processed\train\compound_embedding.npy"
np.save(save_path2,train_embeddings)

print(f"Compound embeddings saved to {save_path}")