import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from .preprocess import extract_faces_from_image, extract_frames_from_video, extract_faces_from_image
import cv2

class DeepfakeModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Inspect input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,224,224]
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess_pil(self, pil_img: Image.Image):
        """
        Convert PIL.Image to NCHW float32 normalized array expected by model.
        Expected normalization: [-1,1] or [0,1] depending on training.
        Adjust here to match training.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3] in 0-1
        # Normalize to mean/std if used during training. Common example:
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        # transpose to [C,H,W]
        arr = arr.transpose(2,0,1).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)  # add batch dim
        return arr

    def predict(self, local_file_path: str):
        """
        Auto-detect filetype (image vs video). Return aggregated result:
        { label: "REAL"/"FAKE", confidence: 0.92, details: [per-frame predictions] }
        """
        ext = os.path.splitext(local_file_path)[1].lower()
        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            return self._predict_video(local_file_path)
        else:
            return self._predict_image(local_file_path)

    def _predict_image(self, local_image_path):
        faces = extract_faces_from_image(local_image_path)
        if len(faces) == 0:
            return {"label": "NO_FACE_DETECTED", "confidence": 0.0, "details": []}
        preds = []
        for face in faces:
            x = self._preprocess_pil(face)
            outputs = self.session.run([self.output_name], {self.input_name: x})
            logits = outputs[0]  # shape (1,2)
            probs = self._softmax(logits[0])
            # assume index 1 == FAKE
            preds.append({"probs": probs.tolist(), "label": "FAKE" if probs[1] > probs[0] else "REAL", "confidence": float(max(probs))})
        # aggregate by average prob
        avg_probs = np.mean([p["probs"] for p in preds], axis=0)
        final_label = "FAKE" if avg_probs[1] > avg_probs[0] else "REAL"
        confidence = float(max(avg_probs))
        return {"label": final_label, "confidence": confidence, "details": preds}

    def _predict_video(self, local_video_path):
        frames = extract_frames_from_video(local_video_path, frame_stride=10)
        if not frames:
            return {"label": "NO_FRAME", "confidence": 0.0, "details": []}
        frame_results = []
        for fpath in frames:
            faces = extract_faces_from_image(fpath)
            # predict for first face only to be fast
            if len(faces) == 0:
                frame_results.append({"frame": fpath, "label": "NO_FACE", "confidence": 0.0})
            else:
                x = self._preprocess_pil(faces[0])
                outputs = self.session.run([self.output_name], {self.input_name: x})
                logits = outputs[0][0]
                probs = self._softmax(logits)
                label = "FAKE" if probs[1] > probs[0] else "REAL"
                frame_results.append({"frame": fpath, "label": label, "confidence": float(max(probs)), "probs": probs.tolist()})
            try:
                os.remove(fpath)
            except:
                pass
        # aggregate over frames
        avg_probs = np.mean([r.get("probs", [1.0, 0.0]) for r in frame_results], axis=0)
        final_label = "FAKE" if avg_probs[1] > avg_probs[0] else "REAL"
        confidence = float(max(avg_probs))
        return {"label": final_label, "confidence": confidence, "details": frame_results}

    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=0)
