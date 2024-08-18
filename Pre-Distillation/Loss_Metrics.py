from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAUROC
from torchmetrics import JaccardIndex, Dice
import torch.nn.functional as F
import torch
import torch.nn as nn

def calculate_metrics_batch(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    # print(((preds == 0) | (preds == 1)).all())
    # print(((targets == 0) | (targets == 1)).all())
    TP = ((preds == 1) & (targets == 1)).float().sum()
    TN = ((preds == 0) & (targets == 0)).float().sum()
    FP = ((preds == 1) & (targets == 0)).float().sum()
    FN = ((preds == 0) & (targets == 1)).float().sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    intersection = TP
    union = TP + FP + FN
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (2 * TP + FP + FN + 1e-6)

    auroc = BinaryAUROC()
    auc = auroc(preds, targets)

    return accuracy.item(), recall.item(), precision.item(), f1.item(), iou.item(), dice.item(), auc.item()

def contrastive_loss(student_features, teacher_features, temperature=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)
    
    student_features_flat = student_features.view(student_features.size(0), -1)  # 展平
    teacher_features_flat = teacher_features.view(teacher_features.size(0), -1)  # 展平
    
    logits = torch.matmul(student_features_flat, teacher_features_flat.T) / temperature
    labels = torch.arange(logits.size(0)).to(device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss
