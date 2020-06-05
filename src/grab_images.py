import cv2
import time
from src.handlers.image_helper import process_webcam_image


def get_images_from_webcam(
    image_dir, camera, name='good', num_images=100, sleep_time_s=0.1
):
    "Captures and saves n images from webcam, waiting 100 ms in between."
    for i in range(num_images):
        _, image = camera.read()
        image = process_webcam_image(image)
        cv2.imwrite(f'{image_dir}/{name}_{i}.jpg', image)
        time.sleep(0.1)


if __name__ == "__main__":
    image_dir = "train_job/images"
    input(
        'please get into a good pose and press enter. Then move around a bit while staying in the good pose.'
    )
    camera = cv2.VideoCapture(0)
    get_images_from_webcam(image_dir, camera=camera, name='good')

    input(
        'please get into a bad pose and press enter. Then move around a bit while staying in the bad pose.'
    )
    get_images_from_webcam(image_dir, camera=camera, name='bad')
    camera.release()
