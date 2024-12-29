from Secondary_Distillation.Loss_Metrics import *
from Secondary_Distillation.Discriminator import *
from Secondary_Distillation.DataLoader import *
from Secondary_Distillation.SegNet_res18 import *
from transformers import SamProcessor
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
from torch.nn import SyncBatchNorm
from torchvision.models import resnet50
import yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

height = config['height']
width = config['width']
output_height = config['output_height']
output_width = config['output_width']
num_classes = config['num_classes']
batch_size = config['batch_size']
lr = config['lr']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam_image_folder = config['image_folder']
sam_mask_folder = config['mask_folder']
sam_test_image_folder = config['test_image_folder']
sam_test_mask_folder = config['test_mask_folder']

def get_data(image_folder, mask_folder, test_image_folder, test_mask_folder, processor_name="sam_base_vit", batch_size=2):
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataloader, test_dataloader

def get_model(model_name = "sam_base_vit",weight = None):
    model = SamModel.from_pretrained(model_name)
    if weight is not None:
        model.load_state_dict(torch.load(weight))

    return model
        

process_name = config['process_name']
model_name = config['sam_model_name']
weight = config['sam_weight']
train_dataloader, test_dataloader = get_data(sam_image_folder=sam_image_folder, sam_mask_folder=sam_mask_folder, sam_test_image_folder=sam_test_image_folder, sam_test_mask_folder=sam_test_mask_folder,batch_size=batch_size,process_name=process_name)
sam_model = get_model(model_name = model_name)
sam_model = sam_model.to(device)

student_model = get_segnet_model()
student_model = student_model.to(device)

def finetune(student_model, sam_model, discriminator, loader, criterion, contrastive_loss_fn, gan_loss, optimizer_G, optimizer_D, device):
    student_model.train()
    discriminator.train()
    running_loss = 0.0
    metrics = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'auc': 0}
    batch_count = 0
    temperature_kd = 2
    tmp_loss = 0
    
    for batch in tqdm(loader, desc="Finetuning"):
        batch_count += 1
        pixel_values = batch['pixel_values'].to(device)
        input_boxes = batch['input_boxes'].to(device)
        ground_truth_mask = batch['ground_truth_mask'].to(device).long()
        ground_truth_mask_resized = nn.functional.interpolate(ground_truth_mask.unsqueeze(1).float(), size=(height, width), mode='bilinear', align_corners=False).squeeze(1).long()
        ground_truth_mask_resized = ground_truth_mask_resized.unsqueeze(1)
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = sam_model(pixel_values = pixel_values,input_boxes=input_boxes,multimask_output=False).pred_masks
            teacher_outputs = teacher_outputs.squeeze(1)
            teacher_outputs_probs = torch.sigmoid(teacher_outputs)
            teacher_outputs_probs = teacher_outputs_probs.cpu().numpy().squeeze()
            teacher_outputs_pred = (teacher_outputs_probs > 0.5).astype(np.uint8)
            teacher_outputs = torch.from_numpy(teacher_outputs_pred).to(device)
            teacher_outputs_gan = teacher_outputs.float().unsqueeze(dim=1)

        
        student_outputs = student_model(pixel_values)
        student_outputs = student_outputs.float()
        student_outputs_resized = nn.functional.interpolate(student_outputs, size=(output_height, output_width), mode='bilinear', align_corners=False)
        max_values, max_indices = torch.max(student_outputs_softmax, dim=1, keepdim=True)
        one_hot = torch.zeros_like(student_outputs_softmax).scatter_(1, max_indices, 1)
        student_outputs_max = (one_hot * student_outputs_softmax).sum(dim=1, keepdim=True)
        student_outputs_max = student_outputs_max.float()
        student_outputs_sigmoid = torch.sigmoid(student_outputs_max / temperature_kd)
        
        # GAN loss with gradient penalty
        real_predictions = discriminator(teacher_outputs_gan.detach())
        fake_predictions = discriminator(student_outputs_max.detach())
        d_loss_real = -torch.mean(real_predictions)
        d_loss_fake = torch.mean(fake_predictions)
        gp = gradient_penalty(discriminator, teacher_outputs_gan, student_outputs_max, device)
        d_loss = d_loss_real + d_loss_fake + 10 * gp
        d_loss.backward()
        optimizer_D.step()

        if batch_count % 5 == 0:
            optimizer_G.zero_grad()
            fake_predictions = discriminator(student_outputs_max)
            g_loss = -torch.mean(fake_predictions) + contrastive_loss_fn(student_outputs_max, teacher_outputs_gan)
            g_loss.backward()
            optimizer_G.step()
        
        preds = torch.argmax(student_outputs, dim=1)
        preds = preds.float()
        preds = nn.functional.interpolate(preds.unsqueeze(1), size=(output_height, output_width), mode='bilinear', align_corners=False)
        preds = preds.float()
        
        batch_metrics = calculate_metrics_batch(preds, ground_truth_mask)
        for i, key in enumerate(metrics.keys()):
            metrics[key] += batch_metrics[i] * pixel_values.size(0)
    
    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
    
    return running_loss / batch_count, metrics

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
            student_outputs_max = torch.argmax(student_outputs_softmax, dim=1, keepdim=True).float()
            loss = criterion(student_outputs, ground_truth_mask_resized) + contrastive_loss_fn(student_outputs_max, teacher_outputs)
            
            running_loss += loss.item() * pixel_values.size(0)
            preds = torch.argmax(student_outputs, dim=1).float()
            preds = nn.functional.interpolate(preds.unsqueeze(1), size=(output_height, output_width), mode='bilinear', align_corners=False)
            
            batch_metrics = calculate_metrics_batch(preds, ground_truth_mask)
            for i, key in enumerate(metrics.keys()):
                metrics[key] += batch_metrics[i] * pixel_values.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
    
    return epoch_loss, metrics

class DummyOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass

in_channels = config['in_channels']
pretrain_epochs = config['pretrain_epochs']
gan_train_epochs = config['gan_train_epochs']
criterion = DiceCELoss(to_onehot_y=True, softmax=True)
discriminator = PretrainedDiscriminator(in_channels=in_channels)
discriminator.to(device)
optimizer_G = optim.Adam(student_model.parameters(), lr=lr)
optimizer_tmp = DummyOptimizer()
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
contrastive_loss_fn = contrastive_loss
# GAN finetune
for epoch in range(gan_train_epochs):
    print(f"GAN Train Epoch {epoch+1}/{gan_train_epochs}")
    
    train_loss, train_metrics,tmp_loss = finetune(student_model, sam_model, discriminator, train_dataloader, criterion, contrastive_loss_fn, gan_loss, optimizer_G, optimizer_D, device)
    val_loss, val_metrics = validate(student_model, sam_model, test_dataloader, criterion, contrastive_loss_fn, device)

    
    print(f"GAN Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(tmp_loss)
    print(f"GAN Train Metrics: Accuracy {train_metrics['accuracy']:.4f}, Recall {train_metrics['recall']:.4f}, Precision {train_metrics['precision']:.4f}, F1 {train_metrics['f1']:.4f}, IoU {train_metrics['iou']:.4f}, Dice {train_metrics['dice']:.4f}, AUC {train_metrics['auc']:.4f}")
    print(f"GAN Val Metrics: Accuracy {val_metrics['accuracy']:.4f}, Recall {val_metrics['recall']:.4f}, Precision {val_metrics['precision']:.4f}, F1 {val_metrics['f1']:.4f}, IoU {val_metrics['iou']:.4f}, Dice {val_metrics['dice']:.4f}, AUC {val_metrics['auc']:.4f}")









