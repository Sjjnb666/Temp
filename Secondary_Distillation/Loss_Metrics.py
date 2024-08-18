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


# WGAN-GP
def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    # 确保 real_data 和 fake_data 的尺寸一致
    if real_data.size() != fake_data.size():
        fake_data = nn.functional.interpolate(fake_data, size=real_data.size()[2:], mode='bilinear', align_corners=False)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    prob_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def contrastive_loss(student_features, teacher_features, temperature=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)
    
    student_features_flat = student_features.view(student_features.size(0), -1)  # flatten
    teacher_features_flat = teacher_features.view(teacher_features.size(0), -1)  # flatten
    
    logits = torch.matmul(student_features_flat, teacher_features_flat.T) / temperature
    labels = torch.arange(logits.size(0)).to(device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss

# GAN_Loss
def gan_loss(predictions, target_is_real):
    target_tensor = torch.full_like(predictions, 1.0) if target_is_real else torch.full_like(predictions, 0.0)
    return nn.BCELoss()(predictions, target_tensor)
