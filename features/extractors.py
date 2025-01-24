# src/features/extractors.py
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
import gc
from fastdtw import fastdtw
from scipy.spatial.distance import cdist

# --------------------------
# Visual Feature Extractor
# --------------------------


class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize InceptionV3 with aux_logits=False
        self.resnet = models.resnet50(pretrained=True)
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)

        # Modify ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Modify Inception
        self.inception.fc = nn.Identity()  # Remove classification layer
        self.inception.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Ensure proper pooling

        # Then disable auxiliary logits, I am doing this here because you can't set it to False on initialization
        self.inception.aux_logits = False
        # del self.inception.AuxLogits
        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False
        self.inception.eval()

    def forward(self, frames):
        if len(frames) == 0:
            return np.zeros(4096, dtype=np.float32)

        # Process in micro-batches
        batch_size = 4  # Reduced for CPU safety
        resnet_feats = []
        inception_feats = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]

            # Preprocess and process ResNet
            resnet_batch = torch.cat([self._preprocess_frame(f) for f in batch])
            with torch.no_grad():
                resnet_out = self.resnet(resnet_batch)
                # Ensure 2D output by squeezing and handling different shapes
                resnet_curr = (
                    resnet_out.squeeze().view(-1, resnet_out.shape[-1]).cpu().numpy()
                )
                resnet_feats.append(resnet_curr)

            # Preprocess and process Inception
            inception_batch = torch.cat([self._preprocess_inception(f) for f in batch])
            with torch.no_grad():
                inception_out = self.inception(inception_batch)
                # Ensure 2D output by squeezing and handling different shapes
                inception_curr = (
                    inception_out.squeeze()
                    .view(-1, inception_out.shape[-1])
                    .cpu()
                    .numpy()
                )
                inception_feats.append(inception_curr)

            # Memory cleanup
            del resnet_batch, inception_batch
            gc.collect()

        # Combine features with dimension validation
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

        return np.concatenate(
            [resnet_all.mean(axis=0), inception_all.mean(axis=0)]  # [2048]  # [2048]
        )  # Final shape [4096]

    # def _preprocess_frame(self, frame):
    #     """ResNet preprocessing"""
    #     frame = cv2.resize(frame, (224, 224))
    #     tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    #     tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
    #         [0.229, 0.224, 0.225]
    #     )
    #     return tensor.unsqueeze(0)

    def _preprocess_frame(self, frame):
        """ResNet preprocessing"""
        # Ensure 3 channels (last-ditch check)
        if frame.shape[-1] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert if needed

        frame = cv2.resize(frame, (224, 224))
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

        # Ensure proper broadcasting for normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Shape [3,1,1]
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # Shape [3,1,1]

        tensor = (tensor - mean) / std  # Now works for [C,H,W]
        return tensor.unsqueeze(0)

    def _preprocess_inception(self, frame):
        """InceptionV3 preprocessing"""
        if frame.shape[-1] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = cv2.resize(frame, (299, 299))
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor / 255.0 - mean) / std

        return tensor.unsqueeze(0)  # [1, C, H, W]

    # def _preprocess_inception(self, frame):
    #     """InceptionV3 preprocessing"""
    #     # Ensure frame is 3-channel
    #     if frame.shape[-1] == 1:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #     elif frame.shape[-1] == 4:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    #     frame = cv2.resize(frame, (299, 299))
    #     print("THIS IS THE FRAME", frame.shape)
    #     # Convert to tensor, ensure float32
    #     tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

    #     # Normalize
    #     tensor = (tensor / 255.0 - 0.5) * 2.0  # Inception-specific normalization
    #     print("THE tensor shape is", tensor.shape)
    #     # Check tensor shape
    #     if tensor.shape != (3, 299, 299):
    #         raise ValueError(f"Preprocessed tensor has invalid shape: {tensor.shape}")
    #     return tensor.unsqueeze(0)


# --------------------------
# Audio Feature Extractor
# --------------------------


class AudioFeatureExtractor(nn.Module):
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.vggish = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.vggish.eval()
        # Freeze parameters
        for param in self.vggish.parameters():
            param.requires_grad = False
        self.mfcc_proj = nn.Linear(40, 128)  # Project 40 MFCC -> 128 dim

    def forward(self, waveform):
        # Convert to tensor and validate
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, T]
        print("WAVEFORM", waveform)
        print()
        print("LENGTH WAVEFORM", len(waveform))
        # Check audio duration
        if len(waveform) < 960:  # 0.96 * 16000Hz = 1536 samples
            return np.zeros(296)
        # Fallback for empty/short audio
        if waveform.numel() == 0:
            return np.zeros((1, 296))  # 2D array

        waveform = waveform.clamp(-1, 1)  # Ensure valid rang
        # try:
        #     # VGGish returns [n_frames, 128]
        vggish_feats = self.vggish(waveform).squeeze(0).cpu().numpy()
        # except Exception as e:
        #     print(f"VGGish failed: {str(e)}")
        #     vggish_feats = np.zeros((1, 128))  # 2D fallback

        # Ensure MFCC/Mel are 2D [time, features]
        mfcc = self._extract_mfcc(waveform.squeeze(0)).T  # [time, 40]
        mel = self._extract_mel(waveform.squeeze(0)).T  # [time, 128]

        # Handle empty features
        # if mfcc.ndim == 1:
        #     mfcc = mfcc.reshape(-1, 1)
        # if mel.ndim == 1:
        #     mel = mel.reshape(-1, 1)

        aligned_mfcc, aligned_mel = self._align_features(mfcc, mel, vggish_feats)
        return np.concatenate(
            [aligned_mfcc.mean(0), aligned_mel.mean(0), vggish_feats.mean(0)]
        )

    def _extract_mfcc(self, waveform):
        mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=40)(waveform)
        mfcc = self.mfcc_proj(mfcc.permute(1, 0))  # [time, 128]
        return mfcc.detach().numpy().reshape(-1, 128)  # Force 2D

    def _extract_mel(self, waveform):
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=128)(
            waveform
        )
        mel = torch.log2(mel + 1e-6)  # Prevent log(0)
        return mel.permute(1, 0).detach().numpy().reshape(-1, 128)

    def _align_features(self, mfcc, mel, vggish):
        """Robust DTW alignment with empty input handling"""
        # Ensure 2D arrays and handle empty inputs

        vggish = np.atleast_2d(vggish)
        mfcc = np.atleast_2d(mfcc)
        mel = np.atleast_2d(mel)

        # Fallback for empty features
        if vggish.size == 0 or mfcc.size == 0 or mel.size == 0:
            return np.zeros(128), np.zeros(128)

        # Align feature dimensions
        feature_dim = min(vggish.shape[1], mfcc.shape[1], mel.shape[1])
        vggish = vggish[:, :feature_dim]
        mfcc = mfcc[:, :feature_dim]
        mel = mel[:, :feature_dim]

        # Ensure temporal consistency
        min_length = min(vggish.shape[0], mfcc.shape[0], mel.shape[0])
        if min_length == 0:
            return np.zeros(feature_dim), np.zeros(feature_dim)

        vggish = vggish[:min_length]
        mfcc = mfcc[:min_length]
        mel = mel[:min_length]

        # Perform alignment with error handling
        try:
            _, path = fastdtw(vggish, mfcc, dist=cdist)
            aligned_mfcc = np.array([mfcc[p[1]] for p in path])
        except Exception as e:
            print(f"MFCC alignment failed: {str(e)}")
            aligned_mfcc = np.zeros_like(mfcc)

        try:
            _, path = fastdtw(vggish, mel, dist=cdist)
            aligned_mel = np.array([mel[p[1]] for p in path])
        except Exception as e:
            print(f"Mel alignment failed: {str(e)}")
            aligned_mel = np.zeros_like(mel)
        print("YELLO", aligned_mel, aligned_mfcc)
        return aligned_mfcc, aligned_mel


# --------------------------
# Main Processing Pipeline
# --------------------------


class AVProcessor:
    def __init__(self):
        self.visual_extractor = VisualFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.sr = self.audio_extractor.sr

    def process_video(self, video_path):
        """Process video and return shot-based features"""
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")

        try:
            # Extract audio
            try:
                self._extract_audio(video_path, audio_path)
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from {video_path}") from e

            # Verify audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio extraction failed for {video_path}")

            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            print(f"Loaded waveform: shape={waveform.shape}, sr={sr}")
            waveform = waveform.mean(dim=0).numpy()
            print(f"Reduced waveform: shape={waveform.shape}")
            print(
                f"Waveform stats: min={waveform.min()}, max={waveform.max()}, mean={waveform.mean()}"
            )

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Detect shots
        shots = self._detect_shots(video_path)

        features = []
        for start_frame, end_frame in shots:
            # Visual features
            frames = self._extract_frames(cap, start_frame, end_frame)
            vis_feats = self.visual_extractor(frames)

            # Audio features
            start_time = start_frame / fps
            end_time = end_frame / fps
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            print(
                f"Shot: frames {start_frame}-{end_frame}, samples {start_sample}-{end_sample}"
            )
            audio_clip = waveform[start_sample:end_sample]
            # import soundfile as sf

            # sf.write(f"clip_{start_frame}_{end_frame}.wav", audio_clip, self.sr)
            # for x in audio_clip:
            #     print(x)
            aud_feats = self.audio_extractor(audio_clip)
            features.append(np.concatenate([vis_feats, aud_feats]))
        print("NEW SHOT IS COMING UP")
        print()
        print()
        cap.release()
        return np.array(features)

    def _extract_audio(self, video_path, audio_path):
        try:
            audio = AudioSegment.from_file(video_path)
            print(
                f"Original audio: duration={len(audio)/1000}s, channels={audio.channels}, frame_rate={audio.frame_rate}"
            )

            audio = audio.set_channels(1)  # Force mono
            audio = audio.set_frame_rate(16000)  # Match VGGish requirements

            print(
                f"Processed audio: duration={len(audio)/1000}s, channels={audio.channels}, frame_rate={audio.frame_rate}"
            )

            audio.export(audio_path, format="wav", bitrate="256k")

            # Verify exported file
            exported_audio = AudioSegment.from_wav(audio_path)
            print(
                f"Exported audio: duration={len(exported_audio)/1000}s, channels={exported_audio.channels}, frame_rate={exported_audio.frame_rate}"
            )
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")

    def _detect_shots(self, video_path):
        """PySceneDetect shot detection"""
        from scenedetect import detect, ContentDetector

        scene_list = detect(video_path, ContentDetector())
        return [(start.get_frames(), end.get_frames()) for (start, end) in scene_list]

    # def _extract_frames(self, cap, start, end):
    #     """Optimized frame extraction with sampling"""
    #     frames = []
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    #     frame_interval = 3  # Only process every 3rd frame
    #     max_frames = 100  # Maximum frames per shot

    #     for _ in range(start, end):
    #         ret, frame = cap.read()
    #         if not ret or len(frames) >= max_frames:
    #             break
    #         if _ % frame_interval == 0:
    #             frames.append(frame)
    #     return frames

    def _extract_frames(self, cap, start, end):
        """Optimized frame extraction with sampling"""
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_interval = 3  # Only process every 3rd frame
        max_frames = 100  # Maximum frames per shot

        for _ in range(start, end):
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
            if _ % frame_interval == 0:
                # Force 3-channel output (works for grayscale/RGBA)
                if frame.shape[-1] == 1:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[-1] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frames.append(frame)
        return frames
