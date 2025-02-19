import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


class KTSDetector:
    def __init__(self, sigma=1.0, min_seg_length=10):
        self.sigma = sigma
        self.min_seg_length = min_seg_length

    def _compute_kernel_matrix(self, features):
        """Compute RBF kernel matrix between feature vectors"""
        dists = cdist(features, features, metric="sqeuclidean")
        K = np.exp(-dists / (2 * self.sigma**2))
        return K

    def detect_shots(self, features):
        """
        Detect shot boundaries using Kernel Temporal Segmentation

        Args:
            features: numpy array of frame features [n_frames x n_features]
        Returns:
            list of shot boundary indices
        """
        n_frames = len(features)

        # Compute kernel matrix
        K = self._compute_kernel_matrix(features)

        # Compute Laplacian
        D = np.diag(np.sum(K, axis=1))
        L = D - K

        # Dynamic programming for optimal segmentation
        costs = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(i + self.min_seg_length, min(n_frames, i + 300)):
                seg_features = features[i : j + 1]
                costs[i, j] = np.trace(
                    seg_features.T @ L[i : j + 1, i : j + 1] @ seg_features
                )

        # Find optimal boundaries
        dp = np.zeros(n_frames)
        back_track = np.zeros(n_frames, dtype=int)

        for t in range(self.min_seg_length, n_frames):
            min_cost = float("inf")
            best_prev = 0

            for s in range(max(0, t - 300), t - self.min_seg_length + 1):
                cost = dp[s] + costs[s + 1, t]
                if cost < min_cost:
                    min_cost = cost
                    best_prev = s

            dp[t] = min_cost
            back_track[t] = best_prev

        # Reconstruct boundaries
        boundaries = []
        t = n_frames - 1
        while t > 0:
            boundaries.append(t)
            t = back_track[t]

        return sorted(boundaries)


class ContentDetector:
    def __init__(self, threshold=27.0, min_scene_len=15):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def _compute_content_val(self, curr_frame, next_frame):
        """
        Compute content value between consecutive frames using HSV color space
        Similar to PySceneDetect's content detector
        """
        # Convert to HSV
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        next_hsv = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)

        # Calculate delta HSV
        delta_hsv = cv2.absdiff(curr_hsv, next_hsv)

        # Weight channels (H=0.5, S=0.25, V=0.25)
        delta_hsv = delta_hsv.astype(np.float32)
        delta_hsv[:, :, 0] *= 0.5
        delta_hsv[:, :, 1] *= 0.25
        delta_hsv[:, :, 2] *= 0.25

        # Average change
        return np.mean(delta_hsv)

    def detect_shots(self, video_path):
        """
        Detect shots using content-aware detection

        Args:
            video_path: Path to video file
        Returns:
            list of frame indices where shots change
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open video file")

        boundaries = []
        frame_num = 0
        last_cut = 0

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return boundaries

        frame_num += 1

        # Process frame pairs
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Compute content difference
            content_val = self._compute_content_val(prev_frame, curr_frame)

            # Check for shot boundary
            if (
                content_val >= self.threshold
                and frame_num - last_cut >= self.min_scene_len
            ):
                boundaries.append(frame_num)
                last_cut = frame_num

            prev_frame = curr_frame
            frame_num += 1

        cap.release()
        return boundaries


def compare_detectors(video_path):
    """
    Compare KTS and Content-aware detection on the same video

    Args:
        video_path: Path to video file
    Returns:
        dict with results from both methods
    """
    # Extract features for KTS
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features (using color histogram for example)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        features.append(hist.flatten())

    cap.release()
    features = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Run KTS
    kts = KTSDetector()
    kts_boundaries = kts.detect_shots(features)

    # Run Content Detector
    content_detector = ContentDetector()
    content_boundaries = content_detector.detect_shots(video_path)

    return {
        "kts_boundaries": kts_boundaries,
        "content_boundaries": content_boundaries,
        "kts_num_shots": len(kts_boundaries),
        "content_num_shots": len(content_boundaries),
    }





# Using Content Detector (PySceneDetect style)
detector = ContentDetector(threshold=27.0)
shots = detector.detect_shots("./Evaluation/SumMe/Air_Force_One.webm.mp4")

# Using KTS
features = extract_features("video.mp4")  # You need to implement feature extraction
kts = KTSDetector()
shots = kts.detect_shots(features)

# Compare both methods
results = compare_detectors("video.mp4")