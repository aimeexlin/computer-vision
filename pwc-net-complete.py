from pwcnet.run import estimate
import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import cv2
from matplotlib import colors
import re
from tqdm import tqdm

root_dir = "./preprocessed_data"
save_folder = "./flow_data"
def list_leaf_folders(root_dir):
    leaf_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:
            leaf_folders.append(dirpath)
    return leaf_folders

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_sorted_images(folder, extensions={'.png', '.jpg', '.jpeg', '.bmp'}):
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in extensions]
    files.sort(key=natural_sort_key)
    return [os.path.join(folder, f) for f in files]

def flow_to_rgb(flow):
    """
    Convert the 2-channel flow output (horizontal and vertical) into an RGB image
    where the pixel color represents the angle and saturation represents the magnitude.
    
    Parameters:
        flow (numpy.ndarray): A numpy array of shape (H, W, 2), where:
            - flow[..., 0] is the horizontal flow (u)
            - flow[..., 1] is the vertical flow (v)
    
    Returns:
        rgb_image (numpy.ndarray): An RGB image (H, W, 3) representing the optical flow
    """

    u = flow[..., 0]
    v = flow[..., 1] 

    magnitude = np.sqrt(u**2 + v**2) 
    direction = np.arctan2(v, u)
    mag_max = np.max(magnitude)
    magnitude = magnitude / mag_max if mag_max != 0 else magnitude

    direction = (direction + np.pi) / (2 * np.pi)
    saturation = magnitude
    value = np.ones_like(magnitude)
    hsv_image = np.stack((direction, saturation, value), axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image.astype(np.float32))
    return rgb_image

def calculate_flow_picture(img1, img2):
    res = estimate(img1, img2)
    res = np.array(res.numpy(force=True).transpose(1, 2, 0), np.float32)
    res = flow_to_rgb(res)
    return res



leaf_folders = list_leaf_folders(root_dir)
for folder in tqdm(leaf_folders):
    relative_path = os.path.relpath(folder, root_dir)
    images = list_sorted_images(folder)
    images = [torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(image))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))) for image in images]
    for i in range(len(images)-1):
        img1, img2 = images[i], images[i+1]
        res = calculate_flow_picture(img1, img2)
        save_path = os.path.join(save_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)
        success = cv2.imwrite(os.path.join(save_path, f"flow_image_{i}.png"), res)





