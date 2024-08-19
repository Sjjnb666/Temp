from Finetune_SAM.DataLoader import *
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
import torch
from torch.optim import Adam
import torch.nn as nn
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import numpy as np
from torchmetrics.classification import BinaryAUROC
import yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

image_folder = config['image_folder']
mask_folder = config['mask_folder']
test_image_folder = config['test_image_folder']
test_mask_folder = config['test_mask_folder']

def get_data(image_folder, mask_folder, test_image_folder, test_mask_folder,process_name="sam_base_vit"):
    large_images = load_image_stack(image_folder)
    test_large_images = load_image_stack(test_image_folder)
    large_masks = load_mask_stack(mask_folder)
    test_large_masks = load_mask_stack(test_mask_folder)
    
    images = make_patches_images(large_images)
    masks = make_patches_masks(large_masks)
    test_images = make_patches_images(test_large_images)
    test_masks = make_patches_masks(test_large_masks)
    
    filtered_images, filtered_masks = make_no_valid(images, masks)
    test_filtered_images, test_filtered_masks = make_no_valid(test_images, test_masks)
    
    dataset = make_dict(filtered_images, filtered_masks)
    test_dataset = make_dict(test_filtered_images, test_filtered_masks)
    
    processor = SamProcessor.from_pretrained(process_name)
    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    test_dataset = SAMDataset(dataset=test_dataset,processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    return train_dataloader, test_dataloader

def get_model(lr = 1e-5,model_name = "sam_base_vit",weight = None):
    model = SamModel.from_pretrained(model_name)
    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model
        


process_name = config['process_name']
batch_size = config['batch_size']
lr = config['lr']
model_name = config['sam_model_name']
weight = config['sam_weight']
train_dataloader, test_dataloader = get_data(image_folder=image_folder, mask_folder=mask_folder, test_image_folder=test_image_folder, test_mask_folder=test_mask_folder,process_name=process_name,batch=batch_size)
model = get_model(lr = 1e-5,model_name = model_name,weight = weight)

optimizer = Adam(model.parameters(), lr=lr)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# Function to calculate IoU
def calculate_iou(predictions, targets):
    intersection = torch.logical_and(targets, predictions).float().sum()
    union = torch.logical_or(targets, predictions).float().sum()
    iou = intersection / (union + 1e-6)
    return iou.item()

# Function to calculate Dice coefficient
def calculate_dice(predictions, targets):
    intersection = torch.sum(predictions * targets)
    dice = (2. * intersection) / (torch.sum(predictions) + torch.sum(targets) + 1e-6)
    return dice.item()

def calculate_metrics_gpu(preds, targets):
    """Calculate accuracy, recall, precision, F1 score, IoU, Dice coefficient, and AUC on GPU"""
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
    preds = preds.float()
    targets = targets.float()

    # Calculate AUC
    auroc = BinaryAUROC()  # Define AUROC metric for binary classification
    auc = auroc(preds, targets)

    return accuracy.item(), recall.item(), precision.item(), f1.item(), iou.item(), dice.item(), auc.item()

# Training loop
num_epochs = config['num_epochs']
tmp = 0.9050

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(train_dataloader):
        # Forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # Compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()
        epoch_losses.append(loss.item())

        all_predictions.append((torch.sigmoid(predicted_masks) > 0.5).detach().flatten())
        all_targets.append(ground_truth_masks.detach().flatten())
    
    # Convert lists to tensors and move to GPU
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics for training set on GPU
    train_accuracy, train_recall, train_precision, train_f1, train_iou, train_dice, train_auc = calculate_metrics_gpu(all_predictions, all_targets)

    # Evaluation on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            test_predictions.append((torch.sigmoid(predicted_masks) > 0.5).detach().flatten())
            test_targets.append(ground_truth_masks.detach().flatten())
    
    # Convert lists to tensors and move to GPU
    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)
    
    # Calculate metrics for test set on GPU
    test_accuracy, test_recall, test_precision, test_f1, test_iou, test_dice, test_auc = calculate_metrics_gpu(test_predictions, test_targets)

    print(f'EPOCH: {epoch + 1}')
    print(f'Mean training loss: {mean(epoch_losses)}')
    print(f'Training Accuracy: {train_accuracy}')
    print(f'Training Recall: {train_recall}')
    print(f'Training Precision: {train_precision}')
    print(f'Training F1 Score: {train_f1}')
    print(f'Training IoU: {train_iou}')
    print(f'Training Dice Coefficient: {train_dice}')
    print(f'Training AUC: {train_auc}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Recall: {test_recall}')
    print(f'Test Precision: {test_precision}')
    print(f'Test F1 Score: {test_f1}')
    print(f'Test IoU: {test_iou}')
    print(f'Test Dice Coefficient: {test_dice}')
    print(f'Test AUC: {test_auc}')
