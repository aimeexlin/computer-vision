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
img2_path = os.path.join(img_path, "frame_10.png")

img1_original = np.array(Image.open(img1_path))[:, :, ::-1]
img1 = torch.FloatTensor(np.ascontiguousarray(img1_original.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
img2 = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img2_path))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

res = estimate(img1, img2)
res = np.array(res.numpy(force=True).transpose(1, 2, 0), np.float32)

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

overlay_image = overlay_optical_flow(img1_original, res, step=6, scale=1)

cv2.imshow('Optical Flow RGB', cv2.resize(overlay_image, (overlay_image.shape[0] * 4, overlay_image.shape[1]*4)))
cv2.waitKey(0)
cv2.destroyAllWindows()