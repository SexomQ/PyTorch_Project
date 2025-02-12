import cv2
import os

from pathlib import Path
from glob import glob
import random
import numpy as np


def convert_images(input_folder, output_folder, n_augments=4):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    input_files = glob(os.path.join(input_folder, "*.png"))
    
    for f in input_files:
        original_image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        
        # Save original image
        original_filename = os.path.basename(f)
        cv2.imwrite(os.path.join(output_folder, original_filename), original_image)
        
        for i in range(n_augments):
            image = original_image.copy()
            
            # Quantization
            image = (image // 43) * 43
            image[image > 43] = 255
            
            # Random rotation (-15 to 15 degrees)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                h, w = image.shape
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Random Gaussian noise
            if random.random() < 0.4:
                noise = np.random.normal(0, random.randint(5, 25), image.shape).astype(np.uint8)
                image = cv2.add(image, noise)
            
            # Random contrast adjustment
            if random.random() < 0.4:
                alpha = random.uniform(0.7, 1.3)  # Contrast factor
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            
            # Save modified image
            filename = os.path.basename(f).replace(".png", f"_aug{i}.png")
            cv2.imwrite(os.path.join(output_folder, filename), image)


if __name__ == "__main__":
    folder_name = "data/train"
    folders = glob(f"{folder_name}/*")
    for f in folders:
        convert_images(f, f.replace(f"{folder_name}", f"{folder_name}_processed"))
