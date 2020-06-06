import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from imageatm.components import Evaluation
from src.handlers.image_helper import process_webcam_image


# load model
class Predictor:
    def __init__(self, image_dir, job_dir):
        e = Evaluation(image_dir=image_dir, job_dir=job_dir)
        self.model = e.best_model

    def predict(self, webcam_image):
        image = process_webcam_image(webcam_image)
        image = preprocess_input(image)
        x = np.expand_dims(image, axis=0)
        result = self.model.predict(x)
        # TODO: convert result to dict referencing class names
        return float(result[0][0])


class Presence_Predictor:
    def __init__(self, presence_image_dir, presence_job_dir):
        e = Evaluation(image_dir=presence_image_dir, job_dir=presence_job_dir)
        self.model = e.best_model

    def predict(self, webcam_image):
        image = process_webcam_image(webcam_image)
        image = preprocess_input(image)
        x = np.expand_dims(image, axis=0)
        result = self.model.predict(x)
        # TODO: convert result to dict referencing class names
        return float(result[0][0])