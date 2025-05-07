from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageShow
import os
import numpy as np
from IPython.display import display
from tqdm import tqdm
import matplotlib.pyplot as plt


# Initialize MTCNN
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')

# Load video
video_path = "./data/manipulated_sequences/Deepfakes/c23/videos/033_097"
folder = f"./preprocessed_data/{video_path}"
cap = cv2.VideoCapture(video_path+".mp4")
frames = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frames.append(frame)  # Frame is a NumPy array (H, W, 3), BGR format

cap.release()


BATCH_SIZE = 256
faces = [frame_faces for i in tqdm(range(0, len(frames), BATCH_SIZE)) for frame_faces in mtcnn(frames[i:min(len(frames),i+BATCH_SIZE)])]

os.makedirs(folder, exist_ok=True)
for i, frame_faces in tqdm(enumerate(faces)):
    face = frame_faces[0]
    img_pil = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
    img_pil.save(os.path.join(folder, f"frame_{i}.png"))