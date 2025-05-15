from DataLoader import *
from SegNet_res18 import *
from Loss_Metrics import *
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from monai.losses import DiceCELoss
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
import cv2
from torchmetrics.classification import BinaryAUROC
import torch.nn.functional as F

height = 1024
width = 1024
output_height = 256
output_width = 256
num_classes = 2
batch_size = 4
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam_image_folder = 'path_to_your_sam_image_folder/'
sam_mask_folder = 'path_to_your_sam_mask_folder/'
sam_test_image_folder = 'path_to_your_sam_test_image_folder/'
sam_test_mask_folder = 'path_to_your_sam_test_mask_folder/'

def get_data(image_folder, mask_folder, test_image_folder, test_mask_folder, processor_name="sam_base_vit"):
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
    
    processor = SamProcessor.from_pretrained(processor_name)
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
        

train_dataloader, test_dataloader = get_data(sam_image_folder, sam_mask_folder, sam_test_image_folder, sam_test_mask_folder)
sam_model = get_model(lr = 1e-5,model_name = "sam_base_vit")
sam_model = sam_model.to(device)

student_model = get_segnet_model()
student_model = student_model.to(device)

# calculate_metrics_batch = Loss_Metrics.calculate_metrics_batch
# contrastive_loss = Loss_Metrics.contrastive_loss
criterion = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

def train(student_model, sam_model, loader, criterion, contrastive_loss_fn, optimizer, device):
    student_model.train()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'auc':0}
    tmp = 0
    
    for batch in tqdm(loader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        input_boxes = batch['input_boxes'].to(device)
        ground_truth_mask = batch['ground_truth_mask'].to(device).long()
        ground_truth_mask_resized = nn.functional.interpolate(ground_truth_mask.unsqueeze(1).float(), size=(height, width), mode='bilinear', align_corners=False).squeeze(1).long()
        ground_truth_mask_resized = ground_truth_mask_resized.unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = sam_model(pixel_values).pred_masks
            teacher_outputs = teacher_outputs[:, :, 0, :, :]
        
        student_outputs = student_model(pixel_values)
        student_outputs_resized = nn.functional.interpolate(student_outputs, size=(output_height, output_width), mode='bilinear', align_corners=False)
        student_outputs_softmax = nn.functional.softmax(student_outputs_resized, dim=1)
        max_values, max_indices = torch.max(student_outputs_softmax, dim=1, keepdim=True)
        one_hot = torch.zeros_like(student_outputs_softmax).scatter_(1, max_indices, 1)
        student_outputs_max = (one_hot * student_outputs_softmax).sum(dim=1, keepdim=True)
        student_outputs_max = student_outputs_max.float()
        loss = criterion(student_outputs, ground_truth_mask_resized) + contrastive_loss_fn(student_outputs_max, teacher_outputs)
        tmp = contrastive_loss_fn(student_outputs_max, teacher_outputs) + tmp
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * pixel_values.size(0)
        preds = torch.argmax(student_outputs, dim=1)

        preds = preds.float()
        preds = nn.functional.interpolate(preds.unsqueeze(1), size=(output_height, output_width), mode='bilinear', align_corners=False)
        preds = preds.float()
        batch_metrics = calculate_metrics_batch(preds, ground_truth_mask)
        for i, key in enumerate(metrics.keys()):
            metrics[key] += batch_metrics[i] * pixel_values.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
    
    return epoch_loss, metrics


def validate(student_model, sam_model, loader, criterion, contrastive_loss_fn, device):
    student_model.eval()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'auc':0}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(device)
            ground_truth_mask = batch['ground_truth_mask'].to(device).long()
            ground_truth_mask_resized = nn.functional.interpolate(ground_truth_mask.unsqueeze(1).float(), size=(height, width), mode='bilinear', align_corners=False).squeeze(1).long()
            ground_truth_mask_resized = ground_truth_mask_resized.unsqueeze(1)
            
            teacher_outputs = sam_model(pixel_values).pred_masks
            teacher_outputs = teacher_outputs[:, :, 0, :, :]
            student_outputs = student_model(pixel_values)
            student_outputs_resized = nn.functional.interpolate(student_outputs, size=(output_height, output_width), mode='bilinear', align_corners=False)
            
            student_outputs_softmax = nn.functional.softmax(student_outputs_resized, dim=1)
            student_outputs_max = torch.argmax(student_outputs_softmax, dim=1, keepdim=True)

            
            student_outputs_max = student_outputs_max.float()
            loss1 = contrastive_loss_fn(student_outputs_max, teacher_outputs)
            loss = criterion(student_outputs, ground_truth_mask_resized) + loss1
            
            running_loss += loss.item() * pixel_values.size(0)
            preds = torch.argmax(student_outputs, dim=1)

            preds = preds.float()
            preds = nn.functional.interpolate(preds.unsqueeze(1), size=(output_height, output_width), mode='bilinear', align_corners=False)
            preds = preds.float()
            batch_metrics = calculate_metrics_batch(preds, ground_truth_mask)
            for i, key in enumerate(metrics.keys()):
                metrics[key] += batch_metrics[i] * pixel_values.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
    
    return epoch_loss, metrics

epochs = 200
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    train_loss, train_metrics = train(student_model, sam_model, train_dataloader, criterion, contrastive_loss, optimizer, device)
    val_loss, val_metrics = validate(student_model, sam_model, test_dataloader, criterion, contrastive_loss, device)
    
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Metrics: Accuracy {train_metrics['accuracy']:.4f}, Recall {train_metrics['recall']:.4f}, Precision {train_metrics['precision']:.4f}, F1 {train_metrics['f1']:.4f}, IoU {train_metrics['iou']:.4f}, Dice {train_metrics['dice']:.4f}, AUC {train_metrics['auc']:.4f}")
    print(f"Val Metrics: Accuracy {val_metrics['accuracy']:.4f}, Recall {val_metrics['recall']:.4f}, Precision {val_metrics['precision']:.4f}, F1 {val_metrics['f1']:.4f}, IoU {val_metrics['iou']:.4f}, Dice {val_metrics['dice']:.4f}, AUC {val_metrics['auc']:.4f}")
