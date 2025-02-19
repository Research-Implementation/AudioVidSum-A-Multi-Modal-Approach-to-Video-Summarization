import os
import tempfile
import shutil
import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models
import cv2
from pydub import AudioSegment
import numpy as np
import soundfile
import gc
from fastdtw import fastdtw
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.inception = models.inception_v3(pretrained=True, aux_logits=True).to(
            self.device
        )

        # Modify architectures
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.inception.fc = nn.Identity()
        self.inception.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.inception.aux_logits = False

        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False
        self.inception.eval()

    def forward(self, frames):
        if len(frames) == 0:
            return np.zeros(4096, dtype=np.float32)

        batch_size = 4
        resnet_feats = []
        inception_feats = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]

            # Process ResNet features
            resnet_batch = torch.cat([self._preprocess_frame(f) for f in batch]).to(
                self.device
            )
            with torch.no_grad():
                resnet_out = self.resnet(resnet_batch)
                resnet_squeezed = resnet_out.squeeze()
                resnet_curr = (
                    resnet_squeezed.view(-1, resnet_squeezed.shape[-1]).cpu().numpy()
                )
                resnet_feats.append(resnet_curr)

            # Process Inception features
            inception_batch = torch.cat(
                [self._preprocess_inception(f) for f in batch]
            ).to(self.device)
            with torch.no_grad():
                inception_out = self.inception(inception_batch)
                inception_squeezed = inception_out.squeeze()
                inception_curr = (
                    inception_squeezed.view(-1, inception_squeezed.shape[-1])
                    .cpu()
                    .numpy()
                )
                inception_feats.append(inception_curr)

            del resnet_batch, inception_batch
            gc.collect()

        resnet_all = (
            np.concatenate(resnet_feats, axis=0)
            if resnet_feats
            else np.zeros((0, 2048))
        )
        inception_all = (
            np.concatenate(inception_feats, axis=0)
            if inception_feats
            else np.zeros((0, 2048))
        )

        return np.concatenate([resnet_all.mean(axis=0), inception_all.mean(axis=0)])

    def _preprocess_frame(self, frame):
        if frame.shape[-1] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, (224, 224))
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)

    def _preprocess_inception(self, frame):
        if frame.shape[-1] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, (299, 299))
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor / 255.0 - mean) / std
        return tensor.unsqueeze(0)


class AudioFeatureExtractor(nn.Module):
    def __init__(self, sr=16000):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr = sr
        self.vggish = torch.hub.load("harritaylor/torchvggish", "vggish").to(
            self.device
        )
        self.vggish.eval()
        for param in self.vggish.parameters():
            param.requires_grad = False
        self.mfcc_proj = nn.Linear(40, 128)
        self.feature_dim = 384  # 128 (VGGish) + 128 (MFCC) + 128 (Mel)

    def forward(self, waveform):
        if waveform is None or len(waveform) < 1:
            return {
                "vggish": np.zeros((1, 128)),
                "mfcc": np.zeros((1, 128)),
                "mel": np.zeros((1, 128)),
            }

        try:
            waveform_tensor = torch.from_numpy(waveform).float().to(self.device)
            if waveform_tensor.ndim == 1:
                waveform_tensor = waveform_tensor.unsqueeze(0)

            # Pad/repeat audio to minimum length
            min_samples = 15360
            if waveform_tensor.shape[1] < min_samples:
                num_repeats = int(np.ceil(min_samples / waveform_tensor.shape[1]))
                waveform_tensor = waveform_tensor.repeat(1, num_repeats)
                waveform_tensor = waveform_tensor[:, :min_samples]

            # Extract features with temporal dimensions
            features = {
                "vggish": self._extract_vggish(waveform_tensor),
                "mfcc": self._extract_mfcc(waveform_tensor),
                "mel": self._extract_mel(waveform_tensor),
            }

            return features

        except Exception as e:
            print(f"Audio processing error: {e}")
            return {
                "vggish": np.zeros((1, 128)),
                "mfcc": np.zeros((1, 128)),
                "mel": np.zeros((1, 128)),
            }

    def _extract_vggish(self, waveform):
        try:
            vggish_input = waveform.squeeze(0).numpy()
            feats = self.vggish(vggish_input, fs=self.sr).cpu().detach().numpy()
            return feats  # [num_windows, 128]
        except Exception as e:
            print(f"VGGish failed: {e}")
            return np.zeros((1, 128))

    def _extract_mfcc(self, waveform):
        try:
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sr,
                n_mfcc=40,
                melkwargs={"n_fft": 2048, "n_mels": 128, "hop_length": 512},
            )(waveform)
            mfcc = self.mfcc_proj(mfcc.permute(0, 2, 1))  # [batch, time, 128]
            return mfcc.squeeze(0).detach().numpy()  # [time, 128]
        except Exception as e:
            print(f"MFCC failed: {e}")
            return np.zeros((1, 128))

    def _extract_mel(self, waveform):
        try:
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_mels=128, n_fft=2048, hop_length=512
            )(waveform)
            mel = torch.log2(mel + 1e-6)
            return mel.squeeze(0).permute(1, 0).detach().numpy()  # [time, 128]
        except Exception as e:
            print(f"Mel failed: {e}")
            return np.zeros((1, 128))


class AVProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.visual_extractor = VisualFeatureExtractor().to(self.device)
        self.audio_extractor = AudioFeatureExtractor().to(self.device)
        self.sr = self.audio_extractor.sr

    def process_video(self, video_path):
        """Process video at 2 frames per second with corresponding audio segments"""
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Extract frames at 2 fps
        frames, frame_times = self._extract_frames(cap, fps, total_frames)
        print(f"Extracted {len(frames)} frames at 2 fps")

        # Extract audio
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")

        try:
            self._extract_audio(video_path, audio_path)
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0).numpy()
        except soundfile.LibsndfileError as load_e:
            print(f"Torchaudio load error (LibsndfileError): {load_e}")
            print(
                "Assuming audio file is corrupted or incompatible. Processing without audio features."
            )
            waveform = None

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Get full audio features
        full_audio_feats = self.audio_extractor(waveform)
        # Align audio features with video frames
        # Align audio features with video frames
        if waveform is not None:
            aligned_audio_feats = self._align_audio_to_video(
                full_audio_feats,
                frame_times,
                audio_length=len(waveform) / self.sr if waveform is not None else 0,
            )
        else:
            aligned_audio_feats = np.zeros(
                (len(frames), self.audio_extractor.feature_dim)
            )

        # Process visual features
        visual_features = [self.visual_extractor([frame]) for frame in frames]
        print(
            "THESE ARE ALIGNED FEATURES",
            aligned_audio_feats,
            len(aligned_audio_feats),
            "DIMENSION",
            aligned_audio_feats.ndim,
        )
        normalized_vis_feat = self._normalize_features(np.array(visual_features))
        normalized_audio_feat = self._normalize_features(aligned_audio_feats)
        return normalized_vis_feat, normalized_audio_feat

    # Min-Max Normalization
    def _normalize_features(self, features):
        normalized = (features - np.min(features)) / (
            np.max(features) - np.min(features)
        )
        return normalized

    def _align_audio_to_video(self, audio_features, frame_times, audio_length):
        """Align all audio feature types to video frames"""
        aligned_feats = []

        # Get feature time grids for each feature type
        time_grids = {
            "vggish": np.arange(audio_features["vggish"].shape[0]) * 0.96,
            "mfcc": np.arange(audio_features["mfcc"].shape[0]) * (512 / self.sr),
            "mel": np.arange(audio_features["mel"].shape[0]) * (512 / self.sr),
        }

        for frame_time in frame_times:
            frame_feats = []

            # Align each feature type separately
            for feat_type in ["vggish", "mfcc", "mel"]:
                feats = audio_features[feat_type]
                times = time_grids[feat_type]

                # Find nearest feature window
                distances = np.abs(times - frame_time)
                nearest_idx = np.argmin(distances)

                # Handle out-of-bounds indices
                nearest_idx = min(max(nearest_idx, 0), feats.shape[0] - 1)
                frame_feats.append(feats[nearest_idx])

            # Concatenate all features for this frame
            aligned_feats.append(np.concatenate(frame_feats))

        return np.array(aligned_feats)

    def _extract_frames(self, cap, fps, total_frames):
        print("THE TOTAL NUMBER OF FRAMES WHERE", total_frames)
        print("THE TOTAL FPS IS", fps)
        """
        Extract frames at 2 fps with corresponding timestamps

        Args:
            cap: VideoCapture object
            fps: Video frame rate
            total_frames: Total number of frames in video

        Returns:
            tuple: (frames list, frame timestamps list)
        """
        frames = []
        frame_times = []

        # Calculate frame interval for 2 fps
        # frame_interval = int(fps / 2)
        # target_fps = 2

        # Calculate total number of frames to extract
        num_frames = int(total_frames)
        print("THE TOTAL NUMBER OF FRAMES IS", num_frames)

        print(f"Extracting {num_frames} frames at {fps} fps")

        for frame_idx in range(num_frames):
            # Calculate target frame position
            target_frame = frame_idx  # * frame_interval

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if ret:
                # Calculate timestamp for this frame
                frame_time = target_frame / fps

                # Basic frame preprocessing
                frame = self._preprocess_frame(frame)

                frames.append(frame)
                frame_times.append(frame_time)
            else:
                print(f"Failed to read frame at position {target_frame}")

            # # Progress indicator
            # if frame_idx % 50 == 0:
            #     print(f"Processed {frame_idx}/{num_frames} frames")

        print(f"Successfully extracted {len(frames)} frames")
        return frames, frame_times

    def _preprocess_frame(self, frame):
        """
        Preprocess frame for feature extraction

        Args:
            frame: Input frame

        Returns:
            processed frame
        """
        try:
            # Handle different color channels
            if frame is None:
                return None

            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[-1] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize for consistency
            frame = cv2.resize(frame, (224, 224))

            # Basic color correction
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=0)

            return frame

        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None

    # def _extract_audio(self, video_path, audio_path):
    #     """Extract audio from video file"""
    #     try:
    #         audio = AudioSegment.from_file(video_path)

    #         # Convert to mono and set sampling rate
    #         audio = audio.set_channels(1)
    #         audio = audio.set_frame_rate(16000)

    #         # Export processed audio
    #         audio.export(audio_path, format="wav", bitrate="256k")

    #         # Verify exported file
    #         if not os.path.exists(audio_path):
    #             raise FileNotFoundError("Audio export failed")

    #     except Exception as e:
    #         print(f"Audio extraction failed: {e}")
    #         raise

    def _extract_audio(self, video_path, audio_path):
        """Extract audio from video file"""
        waveform = None  # Initialize waveform to None
        try:
            audio = AudioSegment.from_file(video_path)

            # Convert to mono and set sampling rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            # Export processed audio
            audio.export(audio_path, format="wav", bitrate="256k")

            # Verify exported file
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio export failed")

            # Load waveform using torchaudio after successful export
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(dim=0).numpy()

        except Exception as e:
            print(f"Audio extraction failed: {e}")
            print(
                "Assuming video has no audio or audio extraction failed. Processing without audio features."
            )
            waveform = None  # Set waveform to None to indicate audio extraction failure

        return waveform

    def process_batch(self, video_paths, batch_size=4):
        """
        Process multiple videos in batches

        Args:
            video_paths: List of video file paths
            batch_size: Number of videos to process in parallel

        Returns:
            dict: Dictionary mapping video paths to their features
        """
        results = {}

        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i : i + batch_size]

            for video_path in batch_paths:
                try:
                    visual_feats, audio_feats = self.process_video(video_path)
                    results[video_path] = {"visual": visual_feats, "audio": audio_feats}
                except Exception as e:
                    print(f"Failed to process {video_path}: {e}")
                    results[video_path] = None

            # Memory cleanup after each batch
            gc.collect()
            torch.cuda.empty_cache()

        return results
