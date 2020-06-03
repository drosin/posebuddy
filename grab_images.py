import cv2
import time

image_dir = 'train_job/images'


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx, :]


def get_images_from_webcam(camera, name='good', n=100):
    for i in range(n):
        _, image = camera.read()
        len_smaller_side = min(image.shape[:2])
        image = crop_center(image, len_smaller_side, len_smaller_side)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'{image_dir}/{name}_{i}.jpg', image)
        time.sleep(0.1)


camera = cv2.VideoCapture(0)
input(
    'please get into a good pose and press enter. Then move around a bit while staying in the good pose.'
)
get_images_from_webcam(camera, name='good')
input(
    'please get into a bad pose and press enter. Then move around a bit while staying in the bad pose.'
)
get_images_from_webcam(camera, name='bad')
camera.release()
