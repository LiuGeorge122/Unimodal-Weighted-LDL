
import torch
import torch.nn.functional as F
import math


lam1 = 0.5  
lam2 = 0.5  

def gaussian_label_distribution(labels_onehot, num_classes=5, sigma=0.5):
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

