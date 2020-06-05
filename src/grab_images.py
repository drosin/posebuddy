import cv2
import time
from src.handlers.image_helper import process_webcam_image


def get_images_from_webcam(
    image_dir, name='good', cam_num=0, num_images=100, sleep_time_s=0.1
):
    "Captures and saves n images from webcam, waiting 100 ms in between."
    camera = cv2.VideoCapture(cam_num)
    for i in range(num_images):
        _, image = camera.read()
        image = process_webcam_image(image)
        cv2.imwrite(f'{image_dir}/{name}_{i}.jpg', image)
        time.sleep(0.1)
    camera.release()


if __name__ == "__main__":
    image_dir = "train_job/images"
    input(
        'please get into a good pose and press enter. Then move around a bit while staying in the good pose.'
    )
    get_images_from_webcam(image_dir, name='good')

    input(
        'please get into a bad pose and press enter. Then move around a bit while staying in the bad pose.'
    )
    get_images_from_webcam(image_dir, name='bad')

    presence_image_dir = "train_job/presence_images"
    input(
        'please sit in your usual work posture and press enter. Just make sure you are in front of the camera.'
    )
    get_images_from_webcam(presence_image_dir, name='present')
    input(
        'please move away from the camera and press enter. This will capture your background for comparison.'
    )
    get_images_from_webcam(presence_image_dir, name='away')