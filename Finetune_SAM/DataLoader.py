import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
import os
import numpy as np
from PIL import Image
from datasets import Dataset
from PIL import Image

patch_size = 256
step = 256


def read_images_and_masks(image_folder, mask_folder):
    # Get all image file names (assuming the image and mask file names are the same)
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    images = []
    masks = []
    
    for image_file in image_files:
        # Construct the full path to the image and mask files
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file) 
        
        # Reading images and masks
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        
        if image is None:
            print(f"Error: Cannot read image file {image_path}")
            continue
        if mask is None:
            print(f"Error: Cannot read mask file {mask_path}")
            continue

        if image.shape[:2] != mask.shape[:2]:
            print(f"Shape mismatch: image {image_file} shape {image.shape}, mask {mask_file} shape {mask.shape}")
            continue
        
        images.append(image)
        masks.append(mask)
    
    try:
        images = np.array(images)
        masks = np.array(masks)

        # If images is a 4D array (N, H, W, C), convert it to (N, H, W)
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images[..., 0]  # Remove the last channel dimension
    except ValueError as e:
        print("Error converting lists to arrays:", e)
        print("Images list length:", len(images))
        print("Masks list length:", len(masks))
        for i, (img, msk) in enumerate(zip(images, masks)):
            if img.shape[:2] != msk.shape[:2]:
                print(f"Shape mismatch at index {i}: image shape {img.shape}, mask shape {msk.shape}")
    
    return images, masks


# Define a function to load images from a directory and resize and stack them into numpy arrays
def load_image_stack(directory, target_size=None, convert_to_rgb=True):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])
    
    images = []
    for file in files:
        with Image.open(file) as img:
            if target_size is None:
                target_size = img.size
            resized_img = img.resize(target_size, Image.LANCZOS)
            if convert_to_rgb:
                resized_img = resized_img.convert("RGB")  # Convert the mask to three channels道
            images.append(np.array(resized_img))
    
    return np.stack(images, axis=0)

def load_mask_stack(directory, target_size=None, convert_to_rgb=True):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])
    
    images = []
    for file in files:
        with Image.open(file) as img:
            if target_size is None:
                target_size = img.size
            resized_img = img.resize(target_size, Image.LANCZOS)
            if convert_to_rgb:
                resized_img = resized_img.convert("L") 
#             print(resized_img.mode)
            images.append(np.array(resized_img))
    
    return np.stack(images, axis=0)

def make_patches_images(large_images):
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = large_images[img]
        patches_img = patchify(large_image, (patch_size, patch_size, 3), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i, j, 0, :, :, :]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)
    return images 

def make_patches_masks(large_masks):
    #Let us do the same for masks
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = large_masks[img]
        patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):

                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches) 
    return masks

def make_no_valid(images, masks):
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]   
    return filtered_images, filtered_masks

def make_dict(filtered_images, filtered_masks):
    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_images],
        "mask": [Image.fromarray(mask) for mask in filtered_masks],
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

from torch.utils.data import Dataset as Datasets
class SAMDataset(Datasets):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["mask"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

    
