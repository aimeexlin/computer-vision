from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageShow
import os
import numpy as np
from IPython.display import display
from tqdm import tqdm
import matplotlib.pyplot as plt


mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')

video_root = './data'
output_root = './preprocessed_data'
BATCH_SIZE = 64


def process_video(path, save_path):
    print(path, save_path)
    cap = cv2.VideoCapture(path)

    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()

    faces = [frame_faces for i in tqdm(range(0, len(frames), BATCH_SIZE)) for frame_faces in mtcnn(frames[i:min(len(frames),i+BATCH_SIZE)])]

    os.makedirs(save_path, exist_ok=True)
    for i, frame_faces in tqdm(enumerate(faces)):
        if frame_faces is None:
            continue
        face = frame_faces[0]
        img_pil = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
        img_pil.save(os.path.join(save_path, f"frame_{i}.png"))


for dirpath, _, filenames in os.walk(video_root):
    for file in filenames:
        if file.endswith('.mp4'):
            video_path = os.path.join(dirpath, file)
            relative_path = os.path.relpath(video_path, video_root)
            output_folder = os.path.join(output_root, os.path.splitext(relative_path)[0])

            process_video(video_path, output_folder)
            