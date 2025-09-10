import onnxruntime as ort
import numpy as np
from PIL import Image

class DeepfakeDetector:
    def __init__(self, model_path="api/models/deepfake_model.onnx"):
        self.session = ort.InferenceSession(model_path)

    def preprocess(self, image: Image.Image):
        image = image.resize((224, 224))       # Example size
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image: Image.Image):
        inputs = {self.session.get_inputs()[0].name: self.preprocess(image)}
        outputs = self.session.run(None, inputs)
        confidence = float(outputs[0][0][0])
        label = "Deepfake" if confidence > 0.5 else "Authentic"
        return {"confidence": confidence, "label": label}
