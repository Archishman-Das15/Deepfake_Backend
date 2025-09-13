
import os
import numpy as np
from PIL import Image
from .preprocess import extract_faces_from_image, extract_frames_from_video
import cv2
import onnxruntime as ort
import tensorflow as tf
import torch
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
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name

        elif ext == ".h5":
            self.model_type = "keras"
            self.model = tf.keras.models.load_model(model_path)

        elif ext == ".pt":
            self.model_type = "torch"
            # Try to load scripted/traced module first, then fallback to regular checkpoint/module
            try:
                self.model = torch.jit.load(model_path, map_location='cpu')
            except Exception:
                loaded = torch.load(model_path, map_location='cpu')
                if isinstance(loaded, torch.nn.Module):
                    self.model = loaded
                elif isinstance(loaded, dict) and "model_state_dict" in loaded:
                    # cannot reconstruct architecture automatically
                    raise ValueError("Loaded checkpoint contains state_dict but no architecture. "
                                     "Provide a scripted/traced .pt or a torch.nn.Module saved with torch.save(model).")
                else:
                    raise ValueError("Unsupported .pt file content. Provide a scripted/traced module or a nn.Module.")
            self.model.to('cpu')
            self.model.eval()

        else:
            raise ValueError("Unsupported model format. Use .onnx, .h5 or .pt")

    def _preprocess_pil(self, pil_img: Image.Image):
        """
        Preprocess for ONNX model: Convert PIL.Image to NCHW float32 normalized array.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = arr.transpose(2,0,1).astype(np.float32)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _preprocess_pil_keras(self, pil_img: Image.Image):
        """
        Preprocess for Keras model: Convert PIL.Image to NHWC float32 normalized array.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _preprocess_pil_torch(self, pil_img: Image.Image):
        """
        Preprocess for PyTorch model: NCHW torch.FloatTensor normalized with common mean/std.
        Adjust normalization/size if your .pt model expects different input.
        """
        img = pil_img.resize((224,224)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = arr.transpose(2,0,1).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,C,H,W]
        return tensor

    def predict(self, local_file_path: str):
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
            try:
                if self.model_type == "onnx":
                    x = self._preprocess_pil(face)
                    outputs = self.session.run([self.output_name], {self.input_name: x})
                    logits = outputs[0]  # shape (1,2) typically
                    logits_np = np.asarray(logits)
                elif self.model_type == "keras":
                    x = self._preprocess_pil_keras(face)
                    logits = self.model.predict(x)
                    logits_np = np.asarray(logits)
                else:  # torch
                    x = self._preprocess_pil_torch(face)
                    with torch.no_grad():
                        out = self.model(x)
                    if isinstance(out, torch.Tensor):
                        logits_np = out.cpu().numpy()
                    elif isinstance(out, (list, tuple)):
                        logits_np = np.asarray(out[0])
                    elif isinstance(out, dict):
                        # try common keys
                        for k in ("logits","output","preds","out"):
                            if k in out:
                                v = out[k]
                                if isinstance(v, torch.Tensor):
                                    logits_np = v.cpu().numpy()
                                else:
                                    logits_np = np.asarray(v)
                                break
                        else:
                            raise ValueError("Unable to interpret model output dict")
                    else:
                        logits_np = np.asarray(out)
            except Exception as e:
                raise RuntimeError(f"Inference failed for face input: {e}")

            # logits_np expected shape (1,2) or (2,)
            if logits_np.ndim == 1:
                arr_logits = logits_np
            else:
                arr_logits = logits_np[0]

            # if already probabilities (sum ~1) skip softmax
            if np.allclose(np.sum(arr_logits), 1.0, atol=1e-3):
                probs = arr_logits
            else:
                probs = self._softmax(arr_logits)

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
                try:
                    if self.model_type == "onnx":
                        x = self._preprocess_pil(faces[0])
                        outputs = self.session.run([self.output_name], {self.input_name: x})
                        logits_np = np.asarray(outputs[0][0])
                    elif self.model_type == "keras":
                        x = self._preprocess_pil_keras(faces[0])
                        logits = self.model.predict(x)[0]
                        logits_np = np.asarray(logits)
                    else:  # torch
                        x = self._preprocess_pil_torch(faces[0])
                        with torch.no_grad():
                            out = self.model(x)
                        if isinstance(out, torch.Tensor):
                            logits_np = out.cpu().numpy()
                            if logits_np.ndim > 1:
                                logits_np = logits_np[0]
                        elif isinstance(out, (list, tuple)):
                            logits_np = np.asarray(out[0])
                        elif isinstance(out, dict):
                            for k in ("logits","output","preds","out"):
                                if k in out:
                                    v = out[k]
                                    if isinstance(v, torch.Tensor):
                                        logits_np = v.cpu().numpy()
                                    else:
                                        logits_np = np.asarray(v)
                                    break
                            else:
                                raise ValueError("Unable to interpret model output dict")
                        else:
                            logits_np = np.asarray(out)
                except Exception as e:
                    raise RuntimeError(f"Inference failed for frame {fpath}: {e}")

                # ensure 1d logits
                if logits_np.ndim > 1:
                    logits = logits_np[0]
                else:
                    logits = logits_np

                if np.allclose(np.sum(logits), 1.0, atol=1e-3):
                    probs = logits
                else:
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