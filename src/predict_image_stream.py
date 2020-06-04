import cv2
import time
import numpy as np
import importlib
from imageatm.components import Evaluation
from src.prediction import Predictor
from src.handlers.config import load_yaml

config_path = 'config/config.yaml'
config = load_yaml(config_path)
predictor = Predictor(config['image_dir'], config['job_dir'])
camera = cv2.VideoCapture(0)
print('to stop, press cntrl+c')
try:
    while True:
        _, image = camera.read()
        probability = predictor.predict(image)
        print(probability)
        time.sleep(0.1)
except KeyboardInterrupt:
    camera.release()
