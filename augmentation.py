import cv2
import numpy as np

def random_crop(image, crop_size):
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_size[0] + 1)
    left = np.random.randint(0, w - crop_size[1] + 1)
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def random_scale(image, scale_range):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def rotation(image, rotation_angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image

def data_augmentation(image, crop_size, scale_range):
    augmented_images = []
    augmented_images.append(random_crop(image, crop_size))
    augmented_images.append(random_scale(image, scale_range))
    augmented_images.append(rotation(image, 90))
    augmented_images.append(rotation(image, 180))
    augmented_images.append(rotation(image, 270))
    return augmented_images