import importlib
import numpy as np
from imageatm.components import Evaluation
from src.handlers.image_helper import process_webcam_image


# load model
class Predictor:
    def __init__(self, image_dir, job_dir):
        e = Evaluation(image_dir=image_dir, job_dir=job_dir)
        self.model = e.best_model
        self.preprocess_mobilenet = importlib.import_module(
            'tensorflow.keras.applications.mobilenet_v2'
        ).preprocess_input

    def predict(self, webcam_image):
        image = process_webcam_image(webcam_image)
        image = self.preprocess_mobilenet(image)
        x = np.expand_dims(image, axis=0)
        result = self.model.predict(x)
        # TODO: convert result to dict referencing class names
        return float(result[0][0])
