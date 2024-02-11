import cv2
import numpy as np
import time
import sys
import multiprocessing
from multiprocessing import Pool
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

            # Convert KeyPoint objects to serializable format
            kp_serializable = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
            keypoints.extend(kp_serializable)

        # Resize image for the next octave
        if octave < num_octaves - 1:
            gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

    return keypoints

def detect_keypoints_for_batch(args):
    images_batch, contrast_threshold = args
    keypoints_batch = []
    for image in images_batch:
        keypoints = detect_keypoints(image, contrast_threshold = contrast_threshold)
        keypoints_batch.append(keypoints)
    return keypoints_batch

if __name__ == '__main__':
    # Load images
    contrast_threshold = 0.02  # Set your contrast threshold here

    images = []
    image_paths = ['images/flower{}.jpg'.format(i) for i in range(7)]
    loaded_images = [cv2.imread(image_path) for image_path in image_paths]
    for image in loaded_images:
        augmented_images = augmentation.data_augmentation(image, (400, 400), (0.25, 3.0))
        images.extend(augmented_images)

    print('Loaded {} images'.format(len(images)))

    # Create a pool of workers
    if len(sys.argv) > 1:
        pool_size = int(sys.argv[1])
    else:
        pool_size = multiprocessing.cpu_count() * 2
    pool = Pool(processes=pool_size)
    print('Using {} processes'.format(pool_size))

    # Split images into batches
    batch_size = round(len(images) / pool_size)
    image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
    print('Split the images into {} batches'.format(len(image_batches)))

    start_time = time.time()
    
    args = [(batch, contrast_threshold) for batch in image_batches]
    keypoints_batches = pool.map(detect_keypoints_for_batch, args)

    # Close the pool of workers
    pool.close()
    pool.join()
    end_time = time.time()

    keypoints_list = [keypoint for keypoints_batch in keypoints_batches for keypoint in keypoints_batch]

    # Draw keypoints on the images
    for i, keypoints in enumerate(keypoints_list):
        image_index = i % len(images)
        keypoints = [cv2.KeyPoint(x=k[0][0], y=k[0][1], size=k[1], angle=k[2], response=k[3], octave=k[4], class_id=k[5]) for k in keypoints]
        image = images[image_index]
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the result
        output_image_path = 'results/flower{}_keypoints.jpg'.format(i)
        cv2.imwrite(output_image_path, image_with_keypoints)

    print('{:.4f}'.format(end_time - start_time))
