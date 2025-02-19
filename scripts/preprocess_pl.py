import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
from features.extractors_im import AVProcessor


class ParallelAVProcessor:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.streams = [torch.cuda.Stream() for _ in range(num_workers)]
        self.lock = Lock()

    def process_video_worker(self, args):
        video_path, output_dir, worker_id = args
        try:
            # Set device and stream for this worker
            torch.cuda.set_device(self.device)
            with torch.cuda.stream(self.streams[worker_id]):
                processor = AVProcessor()

                vid = os.path.splitext(os.path.basename(video_path))[0]
                vid_dir = os.path.join(output_dir, vid)

                # Skip if already processed
                if os.path.exists(vid_dir):
                    if all(
                        [
                            os.path.exists(os.path.join(vid_dir, f))
                            for f in ["visual.npy", "audio.npy"]
                        ]
                    ):
                        return f"Skipped {vid} - already processed"

                # Create output directory
                with self.lock:
                    os.makedirs(vid_dir, exist_ok=True)

                # Process video
                visual_feats, audio_feats = processor.process_video(video_path)

                # Save features
                np.save(
                    os.path.join(vid_dir, "visual.npy"),
                    np.array(visual_feats, dtype=np.float32),
                )
                np.save(
                    os.path.join(vid_dir, "audio.npy"),
                    np.array(audio_feats, dtype=np.float32),
                )

                return f"Successfully processed {vid}"

        except Exception as e:
            return f"Failed to process {vid}: {str(e)}"


def preprocess_dataset_parallel(
    input_dir="Evaluation/TVSum/videos", output_dir="data/processed", num_workers=4
):
    # Initialize parallel processor
    processor = ParallelAVProcessor(num_workers=num_workers)
    os.makedirs(output_dir, exist_ok=True)

    # Get list of videos to process
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    # Prepare arguments for workers
    worker_args = []
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(input_dir, video_file)
        worker_args.append((video_path, output_dir, i % num_workers))

    start_time = time.time()

    # Process videos in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(processor.process_video_worker, args)
            for args in worker_args
        ]

        # Show progress bar
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                print(result)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\nProcessed {len(video_files)} videos in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Set number of workers based on GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    recommended_workers = max(
        1, min(4, gpu_mem // (4 * 1024 * 1024 * 1024))
    )
    print(f"Using {recommended_workers} workers based on available GPU memory")

    preprocess_dataset_parallel(num_workers=recommended_workers)
