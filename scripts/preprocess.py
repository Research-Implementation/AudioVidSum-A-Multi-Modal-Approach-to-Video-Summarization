import os
import cv2
import numpy as np
from pydub import AudioSegment
from scenedetect import detect, ContentDetector
from tqdm import tqdm
from src.features.extractors import AVProcessor


def detect_shots(video_path):
    scene_list = detect(video_path, ContentDetector())
    return [(start.get_frames(), end.get_frames()) for (start, end) in scene_list]


# def extract_shot_features(video_path, shot_boundaries):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     visual_features = []

#     # Extract visual features per shot
#     for start_frame, end_frame in shot_boundaries:
#         frames = []
#         for frame_id in range(start_frame, end_frame):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#             ret, frame = cap.read()
#             if ret:
#                 frames.append(frame)
#         visual_feats = extract_visual_features(frames)  # Use your ResNet50/InceptionV3
#         visual_features.append(visual_feats.mean(axis=0))  # Average over frames

#     # Extract audio features per shot
#     audio = AudioSegment.from_file(video_path)
#     audio_features = []
#     for start_frame, end_frame in shot_boundaries:
#         start_time = start_frame / fps * 1000  # Convert to milliseconds
#         end_time = end_frame / fps * 1000
#         audio_segment = audio[start_time:end_time]
#         audio_feats = extract_audio_features(audio_segment)  # Use MFCC/VGGish
#         audio_features.append(audio_feats.mean(axis=0))

#     return np.array(visual_features), np.array(audio_features)


def preprocess_dataset(input_dir="Evaluation/Test", output_dir="data/processed"):
    processor = AVProcessor()
    # Creating output directory it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    for video_file in tqdm(os.listdir(input_dir)):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_dir, video_file)
        features = processor.process_video(video_path)

        # Save features
        vid = os.path.splitext(video_file)[0]
        np.save(os.path.join(output_dir, f"{vid}.npy"), features)


print("ALREADY RUNNING")
preprocess_dataset()
