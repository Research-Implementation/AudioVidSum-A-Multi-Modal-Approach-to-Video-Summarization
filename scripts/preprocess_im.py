import os
import time
import multiprocessing
import shutil
import numpy as np
from tqdm import tqdm
import concurrent.futures
from src.features.extractors_im import AVProcessor


def process_single_video(video_file, input_dir, output_dir):
    if not video_file.endswith(".mp4"):
        return

    video_path = os.path.join(input_dir, video_file)
    vid = os.path.splitext(video_file)[0]
    vid_dir = os.path.join(output_dir, vid)

    # Skip if already processed
    if os.path.exists(vid_dir):
        if all(
            os.path.exists(os.path.join(vid_dir, f))
            for f in ["visual.npy", "audio.npy"]
        ):
            print(f"Skipping {vid} - already processed")
            return

    os.makedirs(vid_dir, exist_ok=True)

    # Instantiate the processor within the process
    processor = AVProcessor()

    try:
        visual_feats, audio_feats = processor.process_video(video_path)
        np.save(
            os.path.join(vid_dir, "visual.npy"),
            np.array(visual_feats, dtype=np.float32),
        )
        np.save(
            os.path.join(vid_dir, "audio.npy"), np.array(audio_feats, dtype=np.float32)
        )
        print(f"Processed {vid}")
    except Exception as e:
        print(f"Failed to process {vid}: {e}")
        shutil.rmtree(vid_dir, ignore_errors=True)


def preprocess_dataset_parallel(
    input_dir="Evaluation/TVSum/videos", output_dir="data/processed", max_workers=4
):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    start_time = time.time()

    # Use ProcessPoolExecutor to process videos in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_video, video_file, input_dir, output_dir)
            for video_file in video_files
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Video generated an exception: {exc}")

    elapsed_time = time.time() - start_time
    print(
        f"Processed {len(video_files)} videos, Time elapsed: {elapsed_time:.2f} seconds"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Useful on Windows if you're freezing your code
    print("ALREADY RUNNING")
    preprocess_dataset_parallel()
