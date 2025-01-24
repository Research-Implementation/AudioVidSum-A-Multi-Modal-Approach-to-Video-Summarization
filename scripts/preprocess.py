import shutil
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


# def preprocess_dataset(input_dir="Evaluation/Test", output_dir="data/processed"):
#     processor = AVProcessor()
#     # Creating output directory it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     for video_file in tqdm(os.listdir(input_dir)):
#         if not video_file.endswith(".mp4"):
#             continue

#         video_path = os.path.join(input_dir, video_file)
#         features = processor.process_video(video_path)

#         # Save features
#         vid = os.path.splitext(video_file)[0]
#         np.save(os.path.join(output_dir, f"{vid}.npy"), features)


def preprocess_dataset(
    input_dir="Evaluation/TVSum/videos", output_dir="data/processed"
):
    processor = AVProcessor()
    os.makedirs(output_dir, exist_ok=True)

    for video_file in tqdm(os.listdir(input_dir)):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_dir, video_file)
        vid = os.path.splitext(video_file)[0]
        vid_dir = os.path.join(output_dir, vid)

        # Skip if already processed
        if os.path.exists(vid_dir):
            if all(
                [
                    os.path.exists(os.path.join(vid_dir, f))
                    for f in ["visual.npy", "audio.npy"]
                ]
            ):
                print(f"Skipping {vid} - already processed")
                continue

        # Create video-specific directory
        os.makedirs(vid_dir, exist_ok=True)

        try:
            # Process with error handling
            visual_feats, audio_feats = processor.process_video(video_path)
            print("VISUAL FEATS", visual_feats[0].shape[0])
            print("AUDIO FEATS", audio_feats[0].shape[0])
            # Ensure consistent shapes
            if (
                len(visual_feats) == 0
                or len(audio_feats) == 0
                or visual_feats[0].shape[0] != 4096
                or audio_feats[0].shape[0] != 296
            ):
                raise ValueError("Invalid feature dimensions")

            np.save(
                os.path.join(vid_dir, "visual.npy"),
                np.array(visual_feats, dtype=np.float32),
            )
            np.save(
                os.path.join(vid_dir, "audio.npy"),
                np.array(audio_feats, dtype=np.float32),
            )

        except Exception as e:
            print(f"Failed to process {vid}: {str(e)}")
            shutil.rmtree(vid_dir, ignore_errors=True)


print("ALREADY RUNNING")
preprocess_dataset()
