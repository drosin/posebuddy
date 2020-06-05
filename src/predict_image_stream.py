import cv2
import time
import numpy as np
import importlib
from imageatm.components import Evaluation
from src.prediction import Predictor
from src.prediction import Presence_Predictor
from src.handlers.config import load_yaml

config_path = 'config/config.yaml'
config = load_yaml(config_path)
predictor = Predictor(config['image_dir'], config['job_dir'])
presence_predictor = Presence_Predictor(config['presence_image_dir'], config['presence_job_dir'])
camera = cv2.VideoCapture(0)
print('to stop, press cntrl+c')

sleep_time = 1
counter = 0
try:
    while True:
        _, image = camera.read()
        probability = predictor.predict(image)
        presence_probability = presence_predictor.predict(image)

        if presence_probability < 0.5:
            counter+=1
            duration = counter*sleep_time
        else:
            counter = 0
            duration = 0

        if duration > config['present_timer_sec']:
            print("\n You have been sitting for too long, please stand up and walk around!\n")

        print("\nBad posture prob: " + str(round(probability,2)))
        print("Presence prob: " + str(round(1-presence_probability,2)))

        time.sleep(sleep_time)
except KeyboardInterrupt:
    camera.release()
