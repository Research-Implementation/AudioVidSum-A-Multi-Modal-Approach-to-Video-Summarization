import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


class KTS:
    def __init__(self, sigma=1.0, min_seg_length=15, max_seg_length=300):
        self.sigma = sigma
        self.min_seg_length = min_seg_length
        self.max_seg_length = max_seg_length

    def _compute_combined_kernel(self, visual_features, audio_features):
        """
        Compute kernel matrix using both visual and audio features
        """
        # Normalize features
        scaler = StandardScaler()
        visual_norm = scaler.fit_transform(visual_features)

        # For audio features, we need to handle the dictionary structure
        audio_combined = np.concatenate(
            [audio_features["vggish"], audio_features["mfcc"], audio_features["mel"]],
            axis=1,
        )
        audio_norm = scaler.fit_transform(audio_combined)

        # Compute kernel matrices for each modality
        visual_dists = squareform(pdist(visual_norm, metric="sqeuclidean"))
        audio_dists = squareform(pdist(audio_norm, metric="sqeuclidean"))

        # Combine kernels with equal weights
        visual_kernel = np.exp(-visual_dists / (2 * self.sigma**2))
        audio_kernel = np.exp(-audio_dists / (2 * self.sigma**2))

        return 0.5 * (visual_kernel + audio_kernel)

    def segment(self, visual_features, audio_features):
        """
        Perform kernel temporal segmentation using both visual and audio features

        Args:
            visual_features: numpy array from visual.npy [n_frames x 4096]
            audio_features: dict of numpy arrays from audio.npy
                {
                    'vggish': [n_frames x 128],
                    'mfcc': [n_frames x 128],
                    'mel': [n_frames x 128]
                }
        Returns:
            list of frame indices where segments occur
        """
        n_frames = len(visual_features)

        # Compute combined kernel matrix
        K = self._compute_combined_kernel(visual_features, audio_features)

        # Compute normalized Laplacian
        d = np.sum(K, axis=1)
        D = np.diag(d)
        L = D - K

        # Initialize dynamic programming matrices
        costs = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(
                i + self.min_seg_length, min(n_frames, i + self.max_seg_length)
            ):
                # Compute cost for segment [i,j]
                segment_features = visual_features[i : j + 1]
                costs[i, j] = np.trace(
                    segment_features.T @ L[i : j + 1, i : j + 1] @ segment_features
                )

        # Dynamic programming
        dp = np.zeros(n_frames)
        back = np.zeros(n_frames, dtype=int)

        for t in range(self.min_seg_length, n_frames):
            min_cost = float("inf")
            min_idx = 0

            for s in range(
                max(0, t - self.max_seg_length), t - self.min_seg_length + 1
            ):
                cost = dp[s] + costs[s + 1, t]
                if cost < min_cost:
                    min_cost = cost
                    min_idx = s

            dp[t] = min_cost
            back[t] = min_idx

        # Backtrack to find segments
        segments = []
        t = n_frames - 1
        while t > 0:
            segments.append(t)
            t = back[t]

        return sorted(segments)


def process_video_segments(visual_path, audio_path, min_seg_length=15):
    """
    Process pre-extracted features and perform segmentation

    Args:
        visual_path: Path to visual.npy file
        audio_path: Path to audio.npy file
        min_seg_length: Minimum segment length in frames
    Returns:
        list of frame indices where segments occur
    """
    # Load pre-extracted features
    visual_features = np.load(visual_path)
    audio_features = np.load(audio_path, allow_pickle=True).item()

    # Initialize KTS
    kts = KTS(min_seg_length=min_seg_length)

    # Perform segmentation
    segments = kts.segment(visual_features, audio_features)

    return segments


def analyze_segments(segments, fps=25):
    """
    Analyze detected segments

    Args:
        segments: List of frame indices
        fps: Frames per second of the video
    Returns:
        dict with segment statistics
    """
    segment_lengths = np.diff(segments)
    return {
        "num_segments": len(segments),
        "avg_length_frames": np.mean(segment_lengths),
        "avg_length_seconds": np.mean(segment_lengths) / fps,
        "min_length_seconds": np.min(segment_lengths) / fps,
        "max_length_seconds": np.max(segment_lengths) / fps,
    }
