"""
Process the images and create the dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm


# Process the images. Load the images and transform them to bw. Store them in color and bw folders. 
BASE_DIR = '/media/axelrom16/Axel/TrueColor/Data'
DATA_PATH = '/media/axelrom16/Axel/TrueColor/Data/COCO/val2017'
COLOR_PATH = '/media/axelrom16/Axel/TrueColor/Data/color_images'
BW_PATH = '/media/axelrom16/Axel/TrueColor/Data/bw_images'

# Create the directories if they don't exist
os.makedirs(COLOR_PATH, exist_ok=True)
os.makedirs(BW_PATH, exist_ok=True)

image_names = os.listdir(DATA_PATH)


# Function to identify if the image is grayscale
def is_grayscale(image):
    # Check if the image has three color channels
    if len(image.shape) < 3 or image.shape[2] != 3:
        return True  # Image is already in grayscale (single channel)
    
    # Split the image into B, G, R channels
    blue, green, red = cv2.split(image)
    
    # Check if all the channels are the same
    if np.array_equal(blue, green) and np.array_equal(green, red):
        return True  # The image is grayscale
    else:
        return False  # The image is colored
    

def process_images():
    for img in tqdm(image_names, total=len(image_names)):

        # Read the image
        image = cv2.imread(os.path.join(DATA_PATH, img))

        if is_grayscale(image):
            continue

        # Resize the image to 512x512
        image = cv2.resize(image, (512, 512))

        # Convert the image to grayscale
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display the images (optional)
        """
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image)
        axs[0].set_title('Color')
        axs[0].axis('off')
        axs[1].imshow(image_bw, cmap='gray')
        axs[1].set_title('BW')
        axs[1].axis('off')
        plt.show()
        """

        # Save the images
        cv2.imwrite(os.path.join(COLOR_PATH, img), image)
        cv2.imwrite(os.path.join(BW_PATH, img), image_bw)


def create_train_val_split():
    # Read filenames from the directories
    color_images = os.listdir(COLOR_PATH)

    # Create a dataframe
    df = pd.DataFrame(color_images, columns=['filename'])

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # Split the dataframe into train, validation and test sets
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)
    test_size = len(df) - train_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # Save the dataframes
    train_df.to_csv(f'{BASE_DIR}/train.csv', index=False)
    val_df.to_csv(f'{BASE_DIR}/val.csv', index=False)
    test_df.to_csv(f'{BASE_DIR}/test.csv', index=False)


if __name__ == '__main__':
    process_images()
    create_train_val_split()