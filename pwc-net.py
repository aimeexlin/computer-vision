from pwcnet.run import estimate
import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import cv2
from matplotlib import colors
import matplotlib.pyplot as plt



img_path = "./preprocessed_data/manipulated_sequences/DeepFakeDetection/c23/videos/01_02__outside_talking_still_laughing__YVGY8LOK"
img1_path = os.path.join(img_path, "frame_0.png")
img2_path = os.path.join(img_path, "frame_1.png")

img1_original = np.array(Image.open(img1_path))[:, :, ::-1]
img1 = torch.FloatTensor(np.ascontiguousarray(img1_original.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
img2 = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img2_path))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

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
    print(hsv_image)
    rgb_image = colors.hsv_to_rgb(hsv_image.astype(np.float32))
    return rgb_image





res = estimate(img1, img2)
res = np.array(res.numpy(force=True).transpose(1, 2, 0), np.float32)
# res = flow_to_rgb(res)
# print(res)
# print(res.shape)

# Show the result using OpenCV
# cv2.imshow('Optical Flow RGB', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def overlay_optical_flow(frame, flow, step=10, scale=1):
    """
    Overlay optical flow vectors as arrows on the original image.
    
    :param frame: Original image (BGR format)
    :param flow: Optical flow (u, v) of shape (height, width, 2)
    :param step: Step size for skipping flow vectors (to reduce clutter)
    :param scale: Scale factor for the arrows' length
    :return: Image with overlaid optical flow arrows
    """
    flow_image = frame.copy()
    u = flow[..., 0]
    v = flow[..., 1]
    
    mag, ang = cv2.cartToPolar(u, v)
    mag_max = np.max(mag)
    if mag_max != 0:
        mag = mag / mag_max
    
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            dx = int(u[y, x] * scale)
            dy = int(v[y, x] * scale)
            
            color = (0, 255, 0) 
            thickness = 1
            cv2.arrowedLine(flow_image, (x, y), (x + dx, y + dy), color, thickness)

    return flow_image


# Apply the overlay function
overlay_image = overlay_optical_flow(img1_original, res, step=10, scale=15)

cv2.imshow('Optical Flow RGB', cv2.resize(overlay_image, (overlay_image.shape[0] * 3, overlay_image.shape[1]*3)))
cv2.waitKey(0)
cv2.destroyAllWindows()