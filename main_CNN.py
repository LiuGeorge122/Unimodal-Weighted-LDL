# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:06:24 2025
CNN_LDL_5label (适配BERT预处理版本)
@author: LIUSI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, accuracy_score
from torch.optim.lr_scheduler import StepLR
from transformers import get_linear_schedule_with_warmup
import math
# ==================== 配置参数 ====================
SEED = 4321
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 模型参数
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
DROPOUT = 0.5
OUTPUT_DIM = 5  # 5分类
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 16
MAX_TEXT_LEN = 64  # 与BERT保持一致
lam1 = 0.5
lam2 = 0.5
# 训练参数
EPOCHS = 50
LEARNING_RATE = 1e-2
PATIENCE = 10

# ==================== 数据预处理 (与BERT完全一致) ====================
def score_to_label(score):
    """与BERT相同的标签转换函数"""
    if score < 0.2:
        return 0
    elif score < 0.4:
        return 1
    elif score < 0.6:
        return 2
    elif score < 0.8:
        return 3
    else:
        return 4

def load_processed_data(file_path):
    """与BERT相同的原始数据加载方式"""
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            text = lines[i].strip()
            score = float(lines[i+1].strip())
            label = score_to_label(score)
            texts.append(text)
            labels.append(label)
    return pd.DataFrame({'text': texts, 'label': labels})

# 加载数据集
train_df = load_processed_data("./data/data9_label 5 SST-5/train_final.txt")
valid_df = load_processed_data("./data/data9_label 5 SST-5/valid_final.txt")
test_df = load_processed_data("./data/data9_label 5 SST-5/test_final.txt")

# ==================== 文本向量化 (适配CNN) ====================
# 构建词汇表
word_freq = {}
for text in train_df['text']:
    for word in text.split():  # 使用空格分词（与BERT不同但保持简单处理）
        word_freq[word] = word_freq.get(word, 0) + 1

vocab = ['<pad>', '<unk>'] + sorted(word_freq, key=word_freq.get, reverse=True)[:MAX_VOCAB_SIZE-2]
word2idx = {word: idx for idx, word in enumerate(vocab)}

# 加载本地GloVe词向量
glove_embeddings = {}
with open('./glove6b/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

# 构建嵌入矩阵
embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
for i, word in enumerate(vocab):
    if word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

# 文本转索引
def text_to_indices(texts):
    indices = []
    for text in texts:
        tokens = text.split()[:MAX_TEXT_LEN]  # 简单空格分词
        seq = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        seq += [word2idx['<pad>']] * (MAX_TEXT_LEN - len(seq))
        indices.append(seq)
    return torch.LongTensor(indices)

# 处理所有数据集
train_data = text_to_indices(train_df['text'])
valid_data = text_to_indices(valid_df['text'])
test_data = text_to_indices(test_df['text'])

# 创建TensorDataset
train_labels = torch.LongTensor(train_df['label'].values)
valid_labels = torch.LongTensor(valid_df['label'].values)
test_labels = torch.LongTensor(test_df['label'].values)

train_dataset = TensorDataset(train_data, train_labels)
valid_dataset = TensorDataset(valid_data, valid_labels)
test_dataset = TensorDataset(test_data, test_labels)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==================== CNN模型定义 (保持不变) ====================
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, emb_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, emb_dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# ==================== 训练配置 (保持不变) ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
PAD_IDX = word2idx['<pad>']

model = CNNClassifier(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    n_filters=N_FILTERS,
    filter_sizes=FILTER_SIZES,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT,
    pad_idx=PAD_IDX
)

# 加载预训练词向量
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model = model.to(device)

def gaussian_label_distribution(labels, num_classes=5, sigma=0.5):
    """
    将类别索引标签转换为高斯分布
    """
    batch_size = labels.size(0)
    device = labels.device
    
    # 修正：直接使用标签作为中心点，无需argmax
    y_i = labels.float()  # [batch_size]
    
    # 创建类别位置网格 [1, C] -> [batch_size, C]
    j = torch.arange(num_classes, dtype=torch.float32, device=device).unsqueeze(0)  # [1, C]
    j = j.expand(batch_size, -1)  # [batch_size, C]
    
    # 计算高斯核
    scale = 1.0 / (sigma * math.sqrt(2 * math.pi))  # 缩放因子
    exponent = -(j - y_i.unsqueeze(1))**2 / (2 * sigma**2)
    d_i = scale * torch.exp(exponent)  # [batch_size, C]
    
    # 归一化
    S = d_i.sum(dim=1, keepdim=True)
    d_i = d_i / S
    
    return d_i

# 损失函数（保持原有实现）
def kl_loss(inputs, labels):
    """
    输入说明:
    - inputs: 模型输出的概率分布 [batch_size, num_classes]
    - labels: 可以是one-hot编码或类别索引 [batch_size] 或 [batch_size, num_classes]
    """
    # 自动检测标签格式
    if labels.dim() == 1 or labels.size(1) == 1:
        # 类别索引 -> 转为one-hot
        num_classes = inputs.size(1)
        labels_onehot = F.one_hot(labels.long().squeeze(), num_classes).float()
    else:
        # 已经是one-hot编码
        labels_onehot = labels.float()
    
    labels = gaussian_label_distribution(labels)
    #print(labels)
    inputs = torch.clamp(inputs, min=1e-10, max=1.0)
    log_inputs = torch.log(inputs)
    
    return F.kl_div(
        input=log_inputs,
        target=labels,
        reduction='batchmean'
    )

def Weighted_loss(preds, targets):
    N, C = preds.shape
    class_indices = torch.arange(C).float().to(preds.device)
    y_hat = torch.sum(preds * class_indices, dim=1)
    targets = targets.float()  # 直接使用类别索引
    variance = torch.sum(preds * (class_indices.unsqueeze(0) - y_hat.unsqueeze(1))**2, dim=1)
    variance = variance.detach() + 1e-6
    term1 = 0.5 * torch.log(variance)
    term2 = (y_hat - targets)**2 / (2 * variance)
    term3 = 0.5 * torch.log(torch.tensor(2 * torch.pi))
    return torch.mean(term1 + term2 + term3)

def unimodular_loss(preds, targets):
    N, C = preds.shape
    loss = 0.0
    for i in range(N):
        yi = targets[i].item()
        # 处理左侧递增
        for j in range(1, yi):
            if j < C:
                loss += torch.relu(-(preds[i, j] - preds[i, j-1]))
        # 处理右侧递减
        for j in range(yi, C-1):
            loss += torch.relu(-(preds[i, j] - preds[i, j+1]))
    return loss / N



def unimodal_weighted(preds,targets):
    return kl_loss(preds, targets) + lam1*unimodular_loss(preds,targets)+lam2*Weighted_loss(preds, targets)






def compute_ece(probs, labels, num_bins=10):
    """
    计算 Expected Calibration Error (ECE)
    Args:
        probs: 模型输出概率 [N, num_classes]
        labels: 真实标签 [N]
        num_bins: 分箱数
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_size / len(probs)) * np.abs(bin_conf - bin_acc)
    return ece


def train_and_evaluate():
    model = CNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        n_filters=N_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    )
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        print('*',epoch)
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(texts)
            probs = F.softmax(preds, dim=1)
            loss = unimodal_weighted(probs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for texts, labels in valid_loader:
                texts, labels = texts.to(device), labels.to(device)
                out = model(texts)
                pred = torch.argmax(out, dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(val_labels, val_preds)
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_state = model.state_dict()

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 测试集评估
    model.eval()
    test_preds = []
    test_labels = []
    probs_all = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
    acc = accuracy_score(test_labels, test_preds)
    ece = compute_ece(np.array(probs_all), np.array(test_labels), num_bins=10)
    return acc, ece

def run_multiple_times(n_runs=10):
    acc_list = []
    ece_list = []
    for i in range(n_runs):
        print(f"\n====== Run {i+1}/{n_runs} ======")
        acc, ece = train_and_evaluate()
        print(f"Run {i+1} - Accuracy: {acc:.4f}, ECE: {ece:.4f}")
        acc_list.append(acc)
        ece_list.append(ece)
    
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    ece_mean = np.mean(ece_list)
    ece_std = np.std(ece_list)

    print("\n====== Summary ======")
    print(f"Accuracy: Mean = {acc_mean:.4f}, Std = {acc_std:.4f}")
    print(f"ECE: Mean = {ece_mean:.4f}, Std = {ece_std:.4f}")


if __name__ == "__main__":
    run_multiple_times(n_runs=10)



