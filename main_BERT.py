# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:57:21 2024
BERT_LDL
@author: LIUSI
"""
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import math

batch_size = 16
learning_rate = 1e-3
epochs = 20
model_save_path = "./bert_model"
tokenizer_save_path = "./bert_tokenizer"

lam1 = 0.5
lam2 = 1.5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = BertForSequenceClassification.from_pretrained(model_save_path).to(device)
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)


df = pd.read_excel('./data/data1_twitter_financial_news/sent_train.xlsx')

df = df.fillna('')
train_texts = df['text'].tolist()


train_labels_array = np.array(df['label']).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)

labels = encoder.fit_transform(train_labels_array)

input_ids = []
attention_masks = []
for text in train_texts:
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = 64,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
print(input_ids)
print(attention_masks)
labels = torch.tensor(labels)

print(labels)

train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)


df = pd.read_excel('./data/data1_twitter_financial_news/sent_valid.xlsx')
df = df.fillna('')
test_texts = df['text'].tolist()
# val_labels = df['label'].tolist()

valid_labels_array = np.array(df['label']).reshape(-1, 1)

test_labels = encoder.transform(valid_labels_array)

test_input_ids = []
test_attention_masks = []
for text in test_texts:
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = 64,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_labels)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

def gaussian_label_distribution(labels_onehot, num_classes=3, sigma=0.5):
    batch_size = labels_onehot.size(0)
    device = labels_onehot.device
    
    batch_size = labels_onehot.size(0)
    device = labels_onehot.device

    y_i = torch.argmax(labels_onehot, dim=1).float()  

    
    j = torch.arange(num_classes, dtype=torch.float32, device=device).unsqueeze(0)  
    j = j.expand(batch_size, -1)  

    scale = 1.0 / (sigma * math.sqrt(2 * math.pi))  
    exponent = -(j - y_i.unsqueeze(1))**2 / (2 * sigma**2)
    d_i = scale * torch.exp(exponent)  

    S = d_i.sum(dim=1, keepdim=True)  
    d_i = d_i / S

    return d_i

def kl_loss(inputs, labels):
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
    
    targets = torch.argmax(targets, dim=1)

    variance = torch.sum(preds * (class_indices.unsqueeze(0) - y_hat.unsqueeze(1))**2, dim=1)
    variance = variance.detach() + 1e-6  
    term1 = 0.5 * torch.log(variance)
    term2 = (y_hat - targets.float())**2 / (2 * variance)
    term3 = 0.5 * torch.log(torch.tensor(2 * torch.pi))

    
    loss = term1 + term2 + term3

    return sum(loss) / N

def unimodular_loss(preds, targets):
    N, C = preds.shape
    loss = 0
    for i in range(N):
        yi = torch.argmax(targets[i]).item()

        for j in range(1, yi):
            diff = preds[i, j] - preds[i, j-1]
            loss += torch.relu(-diff)  

        for j in range(yi+1, C-1):
            diff = preds[i, j] - preds[i, j+1]
            loss += torch.relu(-diff)
    return loss / N


def unimodal_weighted(preds,targets):
    return kl_loss(preds, targets) + lam1*unimodular_loss(preds,targets)+lam2*Weighted_loss(preds, targets)


def compute_ece(true_labels, pred_probs, n_bins=10):
    
    pred_labels = np.argmax(pred_probs, axis=1)
    confidences = np.max(pred_probs, axis=1)
    
    correct = (pred_labels == true_labels).astype(np.float32)

    
    bin_indices = (confidences * n_bins).astype(int)
    
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_correct = np.bincount(bin_indices, weights=correct, minlength=n_bins)
    bin_conf = np.bincount(bin_indices, weights=confidences, minlength=n_bins)

    
    avg_acc = bin_correct / (bin_counts + 1e-8)  
    avg_conf = bin_conf / (bin_counts + 1e-8)

    
    weights = bin_counts / len(true_labels)
    ece = np.sum(weights * np.abs(avg_acc - avg_conf))

    return ece


    return sum(loss) / N

def NonWeighted_loss(preds, targets): 
    N, C = preds.shape

    class_indices = torch.arange(C).float().to(preds.device)
    y_hat = torch.sum(preds * class_indices, dim=1)

    targets = torch.argmax(targets, dim=1)

    loss = (y_hat - targets.float()) ** 2

    return torch.mean(loss)


best_model_info = {
    'epoch': -1,
    'val_acc': -1,
    'val_loss': float('inf'),
    'state_dict': None,
    'error_samples': [],
    'metrics': {}
}

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm import tqdm
import time


def evaluate_hyperparams(lam1, lam2, epochs=5, save_model=False):
    
    global model
    model = BertForSequenceClassification.from_pretrained(model_save_path).to(device)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dropout = nn.Dropout(p=0.2).to(device)
    mv_loss = MeanVariance()
    torch_loss_fn = nn.CrossEntropyLoss()
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )
    
    
    best_val_acc = 0
    train_time = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        
        model.train()
        train_loss, train_correct = 0, 0
        for batch in train_dataloader:
            input_ids, masks, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=masks)
            logits = outputs.logits
            logits = dropout(logits)
            prob = F.softmax(logits, dim=1)
            
            loss = unimodal_weighted(prob, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_correct += (torch.argmax(prob, dim=1) == torch.argmax(labels, dim=1)).sum().item()
        
        train_time += time.time() - start_time
        
        
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, masks, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=masks)
                logits = outputs.logits
                prob = F.softmax(logits, dim=1)
                val_loss += unimodal_weighted(prob, labels).item()
                val_correct += (torch.argmax(prob, dim=1) == torch.argmax(labels, dim=1)).sum().item()
        
        val_acc = val_correct / len(test_dataset)
        train_acc = train_correct / len(train_dataset)
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_model:
                torch.save(model.state_dict(), f'sensitivity_model_lam1_{lam1}_lam2_{lam2}.pth')
    
    
    return {
        'val_acc': best_val_acc,
        'train_acc': train_acc,
        'train_time': train_time,
        'lam1': lam1,
        'lam2': lam2
    }

from statistics import mean, stdev

def run_grid_experiments():
    lam1_values = [0.5,1,1.5,2]
    lam2_values = [0.5,1,1.5,2]
    num_runs = 10  

    results_summary = []

    for lam1 in lam1_values:
        for lam2 in lam2_values:
            accs = []
            print(f"\n Running for lam1={lam1}, lam2={lam2}")
            for run in range(num_runs):
                print(f"   Run {run+1}/{num_runs}")
                result = evaluate_hyperparams(lam1=lam1, lam2=lam2, epochs=10)
                accs.append(result['val_acc'] * 100)  

            avg = mean(accs)
            std = stdev(accs)
            print(f" lam1={lam1}, lam2={lam2} -> Val Acc = {avg:.2f} ± {std:.2f}")
            results_summary.append({
                "lam1": lam1,
                "lam2": lam2,
                "val_acc_mean": avg,
                "val_acc_std": std
            })

    
    pd.DataFrame(results_summary).to_csv("hyperparam_results_mean_std.csv", index=False)
'''
if __name__ == '__main__':
    run_grid_experiments()
'''


import numpy as np
from sklearn.metrics import precision_score

def run_multiple_experiments(model_class, train_dataloader, test_dataloader, train_dataset, test_dataset,
                             compute_ece, test_texts, device, num_runs=1, epochs=1, learning_rate=1e-3,
                             loss_fn=unimodal_weighted):
    acc_list, precision_list, ece_list = [], [], []

    for run in range(num_runs):
        print(f"\n Run {run + 1}/{num_runs}")
        
        model = model_class().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train_dataloader) * epochs * 0.1),
            num_training_steps=len(train_dataloader) * epochs
        )
        dropout = nn.Dropout(p=0.2).to(device)
        
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            print('*',epoch)
            model.train()
            train_correct = 0

            for batch in train_dataloader:
                input_ids, masks, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=masks)
                logits = dropout(outputs.logits)
                prob = F.softmax(logits, dim=1)

                loss = loss_fn(prob, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_correct += (torch.argmax(prob, dim=1) == torch.argmax(labels, dim=1)).sum().item()

            val_correct = 0
            model.eval()
            predictions, true_labels, probs_all = [], [], []

            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids, masks, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids, attention_mask=masks)
                    logits = outputs.logits
                    prob = F.softmax(logits, dim=1)

                    preds = torch.argmax(prob, dim=1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
                    probs_all.append(prob.cpu().numpy())

                    val_correct += (preds == torch.argmax(labels, dim=1)).sum().item()

            val_acc = val_correct / len(test_dataset)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

        
        model.load_state_dict(best_model_state)
        model.eval()

        
        predictions, true_labels, probs_all = [], [], []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, masks, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=masks)
                prob = F.softmax(outputs.logits, dim=1)
                preds = torch.argmax(prob, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
                probs_all.append(prob.cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        probs_all = np.concatenate(probs_all, axis=0)

        acc = np.mean(predictions == true_labels)
        precision = precision_score(true_labels, predictions, average='macro')
        ece = compute_ece(true_labels, probs_all)

        acc_list.append(acc)
        precision_list.append(precision)
        ece_list.append(ece)

        print(f" Run {run + 1} done | Acc: {acc:.4f}, Precision: {precision:.4f}, ECE: {ece:.4f}")

    
    print("\n Final Results over 10 runs:")
    print(f"Accuracy:  mean={np.mean(acc_list):.4f}, std={np.std(acc_list):.4f}")
    print(f"Precision: mean={np.mean(precision_list):.4f}, std={np.std(precision_list):.4f}")
    print(f"ECE:       mean={np.mean(ece_list):.4f}, std={np.std(ece_list):.4f}")

if __name__ == '__main__':
    from transformers import BertForSequenceClassification



    
    run_multiple_experiments(
        model_class=lambda: BertForSequenceClassification.from_pretrained(model_save_path, num_labels=3),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        compute_ece=compute_ece,
        test_texts=test_texts,
        device=device,
        num_runs=10,
        epochs=15,
        learning_rate=1e-3,
        loss_fn=unimodal_weighted

    )
