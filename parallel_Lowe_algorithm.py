import cv2
import numpy as np
import time
import multiprocessing

def detect_keypoints(image, num_octaves=4, num_scales=5, sigma=1.6, contrast_threshold=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints = []
    pool_size = num_scales
    pool = multiprocessing.Pool(processes=pool_size)
    print('Number of processes: {}'.format(pool_size))
    for octave in range(num_octaves):
        

        scale_levels = [(sigma * 2 ** (scale / num_scales)) for scale in range(num_scales)]
        blurred_list = [cv2.GaussianBlur(gray, (0, 0), sigmaX=scale_level) for scale_level in scale_levels]

        # batch_size = len(blurred_list) // pool_size + 1
        batch_size = round(len(blurred_list) / pool_size)
        blurred_batches = [blurred_list[i:i+batch_size] for i in range(0, len(blurred_list), batch_size)]
        args = [(contrast_threshold, octave, batch) for batch in blurred_batches]

        results = pool.map(keypoints_processor, args)

        keypoints_list = [keypoint for keypoints_batch in results for keypoint in keypoints_batch]
        keypoints_restored = [cv2.KeyPoint(x=k[0][0], y=k[0][1], size=k[1], angle=k[2], response=k[3], octave=k[4], class_id=k[5]) for k in keypoints_list]
        keypoints.extend(keypoints_restored)
        # Resize image for the next octave
        if octave < num_octaves - 1:
            gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

    pool.close()
    pool.join()
    return keypoints

def keypoints_processor(args):
    # print('Process pid: {} started'.format(multiprocessing.current_process().pid))
    sift = cv2.SIFT_create()
    contrast_threshold, octave, batch = args
    for blurred in batch:
        kp = sift.detect(blurred, None)
        if contrast_threshold is not None:
            kp = [k for k in kp if k.response > contrast_threshold]

        # Adjust keypoint coordinates for the octave and scale
        for k in kp:
            k.pt = tuple(np.array(k.pt) * 2 ** octave)
        kp_serializable = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]

    # print('Process pid: {} finish'.format(multiprocessing.current_process().pid))
    return kp_serializable

if __name__ == '__main__':
    # Load images
    image = cv2.imread('images/single_flower.jpg')

    start_time = time.time()
    keypoints = detect_keypoints(image, contrast_threshold=0.04, num_scales=2)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    end_time = time.time()

    output_image_path = 'results/single_flower.jpg'
    cv2.imwrite(output_image_path, image_with_keypoints)
    print('{:.4f}'.format(end_time - start_time))
