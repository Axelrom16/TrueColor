"""
Dataset class
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm
import torch 


class TrueColorDataset:
    def __init__(
            self,
            csv_path,
            color_path,
            bw_path,
            augmentation=False):

        self.data_path = csv_path
        self.color_path = color_path
        self.bw_path = bw_path

        # Read the csv file
        self.data = pd.read_csv(csv_path)
        self.image_names = self.data['filename'].values

    def __len__(self):
        return len(self.image_names)
    
    def augmentation(self, color_image, bw_image):
        # Data augmentation
        # Random flip
        if np.random.rand() < 0.5:
            color_image = np.fliplr(color_image)
            bw_image = np.fliplr(bw_image)
        
        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.randint(0, 360)
            color_image = cv2.rotate(color_image, angle)
            bw_image = cv2.rotate(bw_image, angle)
        
        return color_image, bw_image
    
    def __getitem__(self, idx):
        # Read the images
        color_image = cv2.imread(os.path.join(self.color_path, self.image_names[idx]))
        bw_image = cv2.imread(os.path.join(self.bw_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Normalize the images to [-1, 1]
        color_image = color_image / 127.5 - 1
        bw_image = bw_image / 127.5 - 1

        # Data augmentation (optional)
        #color_image, bw_image = self.augmentation(color_image, bw_image)

        # Transforme to torch tensor
        color_image = torch.tensor(color_image, dtype=torch.float32)
        bw_image = torch.tensor(bw_image, dtype=torch.float32)

        # Add the channel dimension, 3 channels for both images 
        color_image = color_image.permute(2, 0, 1)
        bw_image = torch.cat([bw_image.unsqueeze(0)] * 3, dim=0)

        return color_image, bw_image
    

if __name__ == '__main__':
    dataset = TrueColorDataset(
        csv_path='/media/axelrom16/Axel/TrueColor/Data/train.csv',
        color_path='/media/axelrom16/Axel/TrueColor/Data/color_images',
        bw_path='/media/axelrom16/Axel/TrueColor/Data/bw_images',
        augmentation=False
    )

    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

    # Display the images
    color_image, bw_image = dataset[0]
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(color_image.permute(1, 2, 0))
    axs[0].set_title('Color')
    axs[0].axis('off')
    axs[1].imshow(bw_image.permute(1, 2, 0), cmap='gray')
    axs[1].set_title('BW')
    axs[1].axis('off')
    plt.show()
