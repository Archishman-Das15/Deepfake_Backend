import cv2
import torch
from facenet_pytorch import MTCNN
from typing import List, Tuple
import numpy as np
import os


class FacePreprocessor:
    """
    Handles face detection and preprocessing using MTCNN from facenet-pytorch.
    Designed for both images and video frames.
    """

    def __init__(self, device: str = None, image_size: int = 224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # MTCNN face detector
        self.detector = MTCNN(
            image_size=self.image_size,
            margin=20,
            min_face_size=40,
            device=self.device,
            post_process=True,
            keep_all=False,  # for deepfake detection, usually the primary face is enough
        )

    def preprocess_image(self, img_path: str) -> torch.Tensor:
        """
        Detect and crop the face from an image, return preprocessed tensor.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_tensor = self.detector(img_rgb)

        if face_tensor is None:
            raise ValueError("No face detected in the image")

        return face_tensor.unsqueeze(0)  # shape: (1, 3, H, W)

    def preprocess_video(self, video_path: str, max_frames: int = 16) -> List[torch.Tensor]:
        """
        Sample frames from video and preprocess faces.
        Returns a list of tensors.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // max_frames)

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_tensor = self.detector(frame_rgb)
                if face_tensor is not None:
                    frames.append(face_tensor)

            idx += 1

        cap.release()

        if not frames:
            raise ValueError("No faces detected in video")

        return frames


# ✅ Example usage
if __name__ == "__main__":
    preproc = FacePreprocessor()

    # Test with an image
    try:
        tensor = preproc.preprocess_image("test_face.jpg")
        print("Image preprocessed:", tensor.shape)
    except Exception as e:
        print("Image test failed:", e)

    # Test with a video
    try:
        frames = preproc.preprocess_video("test_video.mp4", max_frames=8)
        print("Video preprocessed:", len(frames), "frames")
    except Exception as e:
        print("Video test failed:", e)
