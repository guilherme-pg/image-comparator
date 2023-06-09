# TO EXECUTE: python app.py

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


def resize_images(img_1, img_2):
    # TO IMPROVE: workaround: the next 2 lines select the smallest values and
    # the 3rd line avoids combining values so that the dimensions do not fail to adjust
    smaller_size_1 = (min(img_1.size[0], img_2.size[0]), min(img_1.size[1], img_2.size[1]))
    smaller_size_2 = (min(img_1.size[0], img_2.size[1]), min(img_1.size[1], img_2.size[0]))
    smaller_size = (min(smaller_size_1[0], smaller_size_2[0]), min(smaller_size_1[1], smaller_size_2[1]))

    image_one = img_1.resize(smaller_size, resample=Image.LANCZOS)
    image_two = img_2.resize(smaller_size, resample=Image.LANCZOS)

    img_1_size = np.array(image_one)
    img_2_size = np.array(image_two)
    img_normal = np.array(img_1)

    cv2.imwrite("static/images/img_normal_size.jpg", img_normal)  # to set as a normal size example
    cv2.imwrite("static/images/img_resized.jpg", img_1_size)  # to set as a resized example

    return img_1_size, img_2_size


# Structural Similarity Index MMeasure (SSIM)
def get_ssim(img_1_size, img_2_size):
    image1, image2 = resize_images(img_1_size, img_2_size)

    # Calculate SSIM score
    score, diff = ssim(image1, image2, win_size=5, full=True, channel_axis=2)

    # The diff image contains the actual image differences between the two images

    cv2.imwrite("static/images/ssim_difference_image.jpg", diff)  # PROBLEM: black image

    return [score]


# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
def get_mse_rmse(img_1_size, img_2_size):
    image1, image2 = resize_images(img_1_size, img_2_size)

    # Get the MSE
    mse = np.mean((image1 - image2) ** 2)

    # Get the RMSE
    rmse = np.sqrt(mse)

    return [mse, rmse]


# Histogram Comparison
def get_histogram_comparison(img_1_size, img_2_size, file_name, figure_title):
    img_1_size = np.array(img_1_size)
    img_2_size = np.array(img_2_size)

    # Calculate the histograms
    hist1 = cv2.calcHist([img_1_size], [2], None, [256], [0, 256])
    hist2 = cv2.calcHist([img_2_size], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # Calculate the histogram similarity using the intersection method
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # TO IMPROVE: automaticamente identificar o maior valor que pode ser Y
    plt.figure()
    plt.title(figure_title)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.plot(hist1, color="red", alpha=0.5, label="Image 1")
    plt.plot(hist2, color="blue", alpha=0.5, label="Image 2")
    plt.legend()
    plt.savefig(f"static/images/{file_name}.jpg")

    return [correlation, chi_square, intersection, bhattacharyya]


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


# Normalized Cross-Correlation (NCC)
def calculate_ncc(img_1_size, img_2_size):
    image1, image2 = resize_images(img_1_size, img_2_size)

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    # Subtract the mean values
    image1 -= mean1
    image2 -= mean2

    # Calculate the standard deviations
    std1 = np.std(image1)
    std2 = np.std(image2)

    # Normalize the images
    image1 /= std1
    image2 /= std2

    # Calculate the cross-correlation
    ncc = np.sum(image1 * image2) / np.sqrt(np.sum(image1**2) * np.sum(image2**2))

    return [ncc]


# Mutual Information (MI)
def calculate_mi(img_1_size, img_2_size, num_bins=256):

    image1, image2 = resize_images(img_1_size, img_2_size)

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Flatten the images to 1D arrays
    flat_image1 = image1.flatten()
    flat_image2 = image2.flatten()

    # Compute the joint histogram
    joint_histogram, _, _ = np.histogram2d(flat_image1, flat_image2, bins=num_bins)

    # Compute the individual histograms
    hist_image1, _ = np.histogram(flat_image1, bins=num_bins)
    hist_image2, _ = np.histogram(flat_image2, bins=num_bins)

    hist_image1 = hist_image1.astype(float)
    hist_image2 = hist_image2.astype(float)

    # Normalize the histograms
    joint_histogram /= np.sum(joint_histogram)
    hist_image1 /= np.sum(hist_image1)
    hist_image2 /= np.sum(hist_image2)

    # Compute the marginal entropies
    entropy_image1 = -np.sum(hist_image1 * np.log2(hist_image1 + np.finfo(float).eps))
    entropy_image2 = -np.sum(hist_image2 * np.log2(hist_image2 + np.finfo(float).eps))

    # Compute the joint entropy
    entropy_joint = -np.sum(joint_histogram * np.log2(joint_histogram + np.finfo(float).eps))

    # Compute the mutual information
    mi = entropy_image1 + entropy_image2 - entropy_joint

    return [mi]

# TO IMPROVE:
# Peak signal-to-noise ratio (PSNR)
# Feature-based similarity index (FSIM)
# Information theoretic-based Statistic Similarity Measure (ISSM)
# Signal to reconstruction error ratio (SRE)
# Spectral angle mapper (SAM)
# Universal image quality index (UIQ)


def process_data(image_1, image_2):
    img_1 = Image.open(image_1)
    img_2 = Image.open(image_2)

    '''
    :parameter img_1, img_2: DATATYPE ->  <class 'werkzeug.datastructures.file_storage.FileStorage'>
    
    Image.open() return a <class 'PIL.JpegImagePlugin.JpegImageFile'>
    '''

    img_control = img_1.copy()

    ssim = get_ssim(img_1, img_2)
    r_mse = get_mse_rmse(img_1, img_2)
    histogram_comparasion = get_histogram_comparison(img_1, img_2, "histogram_comparison", "Histogram Comparison")
    feature_em = get_feature_em(img_1, img_2, "feature_em_matched_image")
    ncc = calculate_ncc(img_1, img_2)
    mi = calculate_mi(img_1, img_2)

    ssim_ctrl = get_ssim(img_1, img_control)
    r_mse_ctrl = get_mse_rmse(img_1, img_control)
    histogram_comparasion_ctrl = get_histogram_comparison(img_1, img_control, "histogram_comparison_control",
                                                          "Histogram Comparison Control")
    feature_em_ctrl = get_feature_em(img_1, img_control, "feature_em_matched_image_control")
    ncc_ctrl = calculate_ncc(img_1, img_control)
    mi_ctrl = calculate_mi(img_1, img_control)

    ssim.extend(ssim_ctrl)
    r_mse.extend(r_mse_ctrl)
    histogram_comparasion.extend(histogram_comparasion_ctrl)
    feature_em.extend(feature_em_ctrl)
    ncc.extend(ncc_ctrl)
    mi.extend(mi_ctrl)

    results = {'ssim': ssim,
               'r_mse': r_mse,
               'hc': histogram_comparasion,
               'fem': feature_em,
               'ncc': ncc,
               'mi': mi
               }

    return results
