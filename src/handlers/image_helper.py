import cv2


def crop_center(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y : start_y + crop_y, start_x : start_x + crop_x, :]


def process_webcam_image(image):
    "Cut to square and bring to mobilenet size of 224*224"
    len_smaller_side = min(image.shape[:2])
    image = crop_center(image, len_smaller_side, len_smaller_side)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return image
