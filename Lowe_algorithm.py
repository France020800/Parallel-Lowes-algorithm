import cv2
import numpy as np
import time
import augmentation

def detect_keypoints(image, num_octaves=4, num_scales=5, sigma=1.6, contrast_threshold=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    keypoints = []
    for octave in range(num_octaves):
        for scale in range(num_scales):
            # Compute the scale level
            scale_level = sigma * 2 ** (scale / num_scales)

            # Apply Gaussian blur to the image
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=scale_level)

            # Detect keypoints at this scale level
            kp = sift.detect(blurred, None)

            # Filter keypoints based on contrast
            if contrast_threshold is not None:
                kp = [k for k in kp if k.response > contrast_threshold]

            # Adjust keypoint coordinates for the octave and scale
            for k in kp:
                k.pt = tuple(np.array(k.pt) * 2 ** octave)

            keypoints.extend(kp)

        # Resize image for the next octave
        if octave < num_octaves - 1:
            gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

    return keypoints

# Multi image version
start_time = time.time()
image_index = 0
for i in range(0, 7):
    image = cv2.imread('images/flower{}.jpg'.format(i))
    augmented_images = augmentation.data_augmentation(image, (400, 400), (0.25, 3.0))
    for augmented_image in augmented_images:
        keypoints = detect_keypoints(augmented_image, contrast_threshold=0.02)

        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(augmented_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the result
        output_image_path = 'results/flower{}_keypoints.jpg'.format(image_index)
        cv2.imwrite(output_image_path, image_with_keypoints)
        image_index += 1

# Single image version
# image = cv2.imread('images/single_flower.jpg')

# start_time = time.time()
# keypoints = detect_keypoints(image, contrast_threshold=0.04, num_scales=2)
# image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# end_time = time.time()

# output_image_path = 'results/single_flower.jpg'
# cv2.imwrite(output_image_path, image_with_keypoints)
# print('{:.4f}'.format(end_time - start_time))
