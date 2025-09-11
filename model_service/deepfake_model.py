import os
import numpy as np
from PIL import Image
from .preprocess import extract_faces_from_image, extract_frames_from_video
import cv2
import onnxruntime as ort
import tensorflow as tf

class DeepfakeModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            self.model_type = "onnx"
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,224,224]
            self.output_name = self.session.get_outputs()[0].name
        elif ext == ".h5":
            self.model_type = "keras"
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Unsupported model format. Use .onnx or .h5")

    def _preprocess_pil(self, pil_img: Image.Image):
        """
        Preprocess for ONNX model: Convert PIL.Image to NCHW float32 normalized array.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3] in 0-1
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = arr.transpose(2,0,1).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)  # add batch dim
        return arr

    def _preprocess_pil_keras(self, pil_img: Image.Image):
        """
        Preprocess for Keras model: Convert PIL.Image to NHWC float32 normalized array.
        Adjust normalization if your .h5 model expects something different.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3] in 0-1
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
            if self.model_type == "onnx":
                x = self._preprocess_pil(face)
                outputs = self.session.run([self.output_name], {self.input_name: x})
                logits = outputs[0]  # shape (1,2)
            else:  # keras
                x = self._preprocess_pil_keras(face)
                logits = self.model.predict(x)  # shape (1,2)
            probs = self._softmax(logits[0])
            preds.append({"probs": probs.tolist(), "label": "FAKE" if probs[1] > probs[0] else "REAL", "confidence": float(max(probs))})
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
            if len(faces) == 0:
                frame_results.append({"frame": fpath, "label": "NO_FACE", "confidence": 0.0})
            else:
                if self.model_type == "onnx":
                    x = self._preprocess_pil(faces[0])
                    outputs = self.session.run([self.output_name], {self.input_name: x})
                    logits = outputs[0][0]
                else:  # keras
                    x = self._preprocess_pil_keras(faces[0])
                    logits = self.model.predict(x)[0]
                probs = self._softmax(logits)
                label = "FAKE" if probs[1] > probs[0] else "REAL"
                frame_results.append({"frame": fpath, "label": label, "confidence": float(max(probs)), "probs": probs.tolist()})
            try:
                os.remove(fpath)
            except:
                pass
        avg_probs = np.mean([r.get("probs", [1.0, 0.0]) for r in frame_results], axis=0)
        final_label = "FAKE" if avg_probs[1] > avg_probs[0] else "REAL"
        confidence = float(max(avg_probs))
        return {"label": final_label, "confidence": confidence, "details": frame_results}

    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=0)