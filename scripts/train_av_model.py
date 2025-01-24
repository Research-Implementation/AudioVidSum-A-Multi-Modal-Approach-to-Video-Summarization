import torch
from torch.utils.data import DataLoader
from models.av_model import AVBiLSTMModel
from src.data.dataset import TVSumDataset
import pandas as pd
from src.utils.alignments import align_shots_to_annotations
import h5py
import torch.nn.functional as F


def train():
    # Load MATLAB annotations
    with h5py.File(
        "Evaluation/TVSum/ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat", "r"
    ) as f:
        # ======================================================================
        # 1. Get Video Metadata (unchanged)
        # ======================================================================
        titles_ref = f["tvsum50/title"][:]
        videos_ref = f["tvsum50/video"][:]
        categories_ref = f["tvsum50/category"][:]

        titles = [
            "".join(chr(c) for c in f[ref][:].flatten()) for ref in titles_ref.squeeze()
        ]
        videos = [
            "".join(chr(c) for c in f[ref][:].flatten()) for ref in videos_ref.squeeze()
        ]
        categories = [
            "".join(chr(c) for c in f[ref][:].flatten())
            for ref in categories_ref.squeeze()
        ]

        lengths = f["tvsum50/length"][:].flatten()
        nframes = f["tvsum50/nframes"][:].flatten()

        # ======================================================================
        # 2. Correct User Annotation Handling (Key Fix)
        # ======================================================================
        user_anno = f["tvsum50/user_anno"][:]  # Shape: (50, 1) - not (50, 20)!

        annotation_data = []
        for vid_idx in range(50):  # 50 videos
            # Get reference to this video's user annotations (shape: 20 × frames)
            video_user_refs = user_anno[vid_idx, 0]
            user_annotations = f[video_user_refs][:]  # Shape: (20, n_frames)

            for user_idx in range(20):  # 20 users per video
                annotation_data.append(
                    {
                        "Video Title": titles[vid_idx],
                        "Video File Name": videos[vid_idx],
                        "User ID": user_idx + 1,
                        "Annotations": user_annotations[user_idx].flatten(),
                    }
                )

        df_annotations = pd.DataFrame(annotation_data)

    # Create dataset
    dataset = TVSumDataset(df_annotations, "data/processed")

    # Create dataloader with custom collate
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x[0])

    # Model and optimizer
    model = AVBiLSTMModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(100):
        model.train()
        for features, frame_scores in loader:
            # Get shot boundaries from features
            num_shots = features["visual"].shape[0]

            # Align frame scores to shots
            shot_scores = align_shots_to_annotations(
                shot_boundaries=[
                    (0, num_shots)
                ],  # Modify based on actual shot boundaries
                annotations=frame_scores.numpy(),
                fps=30,  # Get actual FPS from video metadata
            )

            # Forward pass
            visual = features["visual"].unsqueeze(0).cuda()
            audio = features["audio"].unsqueeze(0).cuda()
            preds = model(visual, audio)

            # Loss calculation
            loss = F.mse_loss(preds, shot_scores.cuda())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
