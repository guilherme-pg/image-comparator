'''
TO IMPLEMENT
signature comparator
'''

# TO EXECUTE: python app.py

import cv2
import numpy as np
from PIL import Image
from skimage import measure




def resize_images(img_1, img_2):
    '''
    resizes two images to their smallest common size, 
    saving a resized version and the original version of one of the images.
    '''
    # select the smallest values
    new_size = (min(img_1.width, img_2.width), min(img_1.height, img_2.height))
    image_one = img_1.resize(new_size, Image.Resampling.LANCZOS)
    image_two = img_2.resize(new_size, Image.Resampling.LANCZOS)

    img_1_size = np.array(image_one)
    img_2_size = np.array(image_two)
    img_normal = np.array(img_1)

    cv2.imwrite("static/images/img_normal_size.jpg", img_normal)  # to set as a normal size example
    cv2.imwrite("static/images/img_resized.jpg", img_1_size)  # to set as a resized example

    return img_1_size, img_2_size


# Structural Similarity Index MMeasure (SSIM)
# MANTER
def get_ssim(img_1_size, img_2_size):
    '''
        Evaluates how structurally similar two images are.
    '''
    image1, image2 = resize_images(img_1_size, img_2_size)

    # Calculate SSIM score
    score, diff = measure.compare_ssim(image1, image2, win_size=5, full=True, channel_axis=2)


    # The diff image contains the actual image differences between the two images

    diff_uint8 = (diff * 255).astype("uint8")
    cv2.imwrite("static/images/ssim_difference_image.jpg", diff_uint8)
    #cv2.imwrite("static/images/ssim_difference_image.jpg", diff)  # PROBLEM: black image

    return [score]


# Feature Extraction and Matching
def get_feature_em(img_1, img_2, file_name):
    # Initialize the feature extractor (SIFT in this case)
    sift = cv2.SIFT_create()

    img_1 = np.array(img_1)
    img_2 = np.array(img_2)

    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)

    # Detect and compute keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img_1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_2, None)

    # Initialize the feature matcher (Brute-Force)
    matcher = cv2.BFMatcher()

    # Match the descriptors of the two images
    matches = matcher.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute the similarity score based on the number of matches
    similarity = len(matches)

    matched_image = cv2.drawMatches(img_1, keypoints1, img_2, keypoints2, matches[:50], None)

    cv2.imwrite(f"static/images/{file_name}.jpg", matched_image)

    return [similarity]


def process_data(image_1, image_2):
    img_1 = Image.open(image_1)
    img_2 = Image.open(image_2)

    img_control = img_1.copy()

    ssim = get_ssim(img_1, img_2)
    feature_em = get_feature_em(img_1, img_2, "feature_em_matched_image")
    
    ssim_ctrl = get_ssim(img_1, img_control)
    feature_em_ctrl = get_feature_em(img_1, img_control, "feature_em_matched_image_control")

    ssim.extend(ssim_ctrl)
    feature_em.extend(feature_em_ctrl)

    results = {'ssim': ssim,
               'fem': feature_em
               }

    return results
