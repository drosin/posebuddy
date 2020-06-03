import cv2
import time
import numpy as np
import importlib
from imageatm.components import Evaluation

job_dir = 'train_job'
image_dir = 'train_job/images'
json_file_name = 'data.json'


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx, :]


# load model
e = Evaluation(image_dir=image_dir, job_dir=job_dir)
e._make_prediction_on_test_set()
model = e.best_model
preprocess = importlib.import_module(
    'tensorflow.keras.applications.mobilenet_v2'
).preprocess_input

# start image stream and run it until ctrl+c is pressed
camera = cv2.VideoCapture(0)
print('to stop, press cntrl+c')
try:
    while True:
        _, image = camera.read()
        len_smaller_side = min(image.shape[:2])
        image = crop_center(image, len_smaller_side, len_smaller_side)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image = preprocess(image)
        x = np.expand_dims(image, axis=0)
        result = model.predict(x)
        print(result)
        time.sleep(0.1)
except KeyboardInterrupt:
    camera.release()
