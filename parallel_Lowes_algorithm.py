import cv2
import numpy as np
import multiprocessing
import sys
import time

def detect_keypoints_async(image, num_octaves=4, num_scales=5, sigma=1.6, contrast_threshold=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    keypoints = []
    iter = 0
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

def parallel_detect_keypoints(images):
    keypoints_list = []
    for image in images:
        keypoints = detect_keypoints_async(image, contrast_threshold=0.02)
        keypoints_list.append(keypoints)
    
    return keypoints_list

def draw_keypoints(image, keypoints_serializable):
    kp_data = keypoints_serializable[0]
    keypoints = [cv2.KeyPoint(x=k[0][0], y=k[0][1], size=k[1], angle=k[2], response=k[3], octave=k[4], class_id=k[5]) for k in kp_data]
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_keypoints


if __name__ == '__main__':
    # Create multiprocessing pool
    if len(sys.argv) > 1:
        pool_size = int(sys.argv[1])
    else:
        pool_size = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(processes=pool_size)
    print('Using {} processes'.format(pool_size))

    # Load images
    image_paths = ['images/flower{}.jpg'.format(i) for i in range(16)]
    images = [cv2.imread(image_path) for image_path in image_paths]

    # Split the images into chunks
    image_chunks = [images[i::pool_size] for i in range(pool_size)]

    # Process images in parallel
    start_time = time.time()
    results = pool.map(parallel_detect_keypoints, image_chunks)
    end_time = time.time()

    # Draw keypoints on the images
    for i, keypoints in enumerate(results):
        if i == len(images):
            break

        image_with_keypoints = draw_keypoints(images[i], keypoints)

        # Save the result
        output_image_path = 'results/flower{}_keypoints.jpg'.format(i)
        cv2.imwrite(output_image_path, image_with_keypoints)

    print('{:.4f}'.format(end_time - start_time))