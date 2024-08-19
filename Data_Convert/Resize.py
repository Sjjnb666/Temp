import os
import cv2
import glob
import argparse

def resize_images_in_folder(folder_path, target_size=(1024, 1024)):
    image_paths = glob.glob(os.path.join(folder_path, '*.*'))
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        resized_image = cv2.resize(image, target_size)
        
        cv2.imwrite(image_path, resized_image)
        print(f"Resized and saved {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a folder to a specified size.")

    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--size", type=int, nargs=2, default=[1024, 1024], help="Target size for resizing images, e.g., --size 1024 1024.")

    args = parser.parse_args()

    resize_images_in_folder(args.folder, tuple(args.size))
