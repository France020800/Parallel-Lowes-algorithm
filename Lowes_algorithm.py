import cv2
import numpy as np

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

# Load images
for i in range(0, 12):
    image = cv2.imread('images/flower{}.jpg'.format(i))
    keypoints = detect_keypoints(image, contrast_threshold=0.02)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the result
    #cv2.imshow('Image with keypoints {}'.format(i), image_with_keypoints)

    # Save the result
    output_image_path = 'results/flower{}_keypoints.jpg'.format(i)
    cv2.imwrite(output_image_path, image_with_keypoints)
