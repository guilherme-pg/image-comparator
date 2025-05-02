# TO EXECUTE: python app.py

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import metrics



def resize_images(img_1, img_2):
    '''
    resizes two images to their smallest common size, 
    saving a resized version and the original version of one of the images.
    '''
    
    # select the smallest values
    new_size = (min(img_1.width, img_2.width), min(img_1.height, img_2.height))
    image_one = img_1.resize(new_size, Image.Resampling.LANCZOS)
    image_two = img_2.resize(new_size, Image.Resampling.LANCZOS)

    img_1_size = cv2.cvtColor(np.array(image_one), cv2.COLOR_RGB2BGR)
    img_2_size = cv2.cvtColor(np.array(image_two), cv2.COLOR_RGB2BGR)
    # img_1_normal = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
    # img_2_normal = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2BGR)

    # cv2.imwrite("static/images/img_1_normal_size.jpg", img_1_normal)  # to set as a normal size example
    cv2.imwrite("static/images/img_1_resized.jpg", img_1_size)  # to set as a resized example
    # cv2.imwrite("static/images/img_2_normal_size.jpg", img_2_normal)  # to set as a normal size example
    cv2.imwrite("static/images/img_2_resized.jpg", img_2_size)  # to set as a resized example

    return img_1_size, img_2_size



# Structural Similarity Index MMeasure (SSIM)
def get_ssim(img_1_size, img_2_size):
    '''
        Evaluates how structurally similar two images are.
    '''
    image1, image2 = resize_images(img_1_size, img_2_size)

    # Calculate SSIM score
    score, diff = metrics.structural_similarity(image1, image2, win_size=5, full=True, channel_axis=2)

    # The diff image contains the actual image differences between the two images

    diff_uint8 = (diff * 255).astype("uint8")
    cv2.imwrite("static/images/ssim_difference_image.jpg", diff_uint8)
    

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
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.savefig(f"static/images/{file_name}.jpg")

    return [correlation, chi_square, intersection, bhattacharyya]


# Feature Extraction and Matching
def get_feature_em(img_1, img_2, file_name):
    # Inicializa o detector SIFT
    sift = cv2.SIFT_create()

    # Converte imagens para escala de cinza
    img_1 = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2GRAY)

    # Detecta keypoints e descritores
    keypoints1, descriptors1 = sift.detectAndCompute(img_1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_2, None)

    # Verifica se há descritores
    if descriptors1 is None or descriptors2 is None:
        return [0.0]

    # Inicializa o matcher Brute-Force
    matcher = cv2.BFMatcher()

    # Aplica o teste de Lowe com k=2
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Similaridade normalizada pela quantidade de keypoints
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))

    # Salva imagem com os 50 melhores matches
    matched_image = cv2.drawMatches(img_1, keypoints1, img_2, keypoints2, good_matches[:50], None)
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


def classify_similarity(value, reference, metric_name):
    if metric_name == 'ssim':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.01:
            return 'very similar'
        elif diff < 0.05:
            return 'relatively similar'
        elif diff < 0.1:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'r_mse':
        normalized_mse = value / (255 ** 2)
        if normalized_mse == 0:
            return 'equals'
        elif normalized_mse < 0.001:
            return 'very similar'
        elif normalized_mse < 0.01:
            return 'relatively similar'
        elif normalized_mse < 0.05:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'r_rmse':
        normalized_rmse = value / 255
        if normalized_rmse == 0:
            return 'equals'
        elif normalized_rmse < 0.02:
            return 'very similar'
        elif normalized_rmse < 0.06:
            return 'relatively similar'
        elif normalized_rmse < 0.12:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'hc_correlation':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.01:
            return 'very similar'
        elif diff < 0.05:
            return 'relatively similar'
        elif diff < 0.1:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'hc_chi_square':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.2:
            return 'very similar'
        elif diff < 0.6:
            return 'relatively similar'
        elif diff < 1.0:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'hc_intersection':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.1:
            return 'very similar'
        elif diff < 0.3:
            return 'relatively similar'
        elif diff < 0.5:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'hc_bhattacharyya':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.1:
            return 'very similar'
        elif diff < 0.3:
            return 'relatively similar'
        elif diff < 0.5:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'fem':
        if reference == 0:
            return 'very different'
        ratio = value / reference
        if ratio == 1.0:
            return 'equals'
        elif ratio >= 0.9:
            return 'very similar'
        elif ratio >= 0.7:
            return 'relatively similar'
        elif ratio >= 0.5:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'ncc':
        diff = abs(value - reference)
        if diff == 0:
            return 'equals'
        elif diff < 0.01:
            return 'very similar'
        elif diff < 0.05:
            return 'relatively similar'
        elif diff < 0.1:
            return 'not very similar'
        else:
            return 'very different'

    elif metric_name == 'mi':
        if not reference:
            return 'undefined'
        ratio = value / reference
        if ratio == 1.0:
            return 'equals'
        elif ratio >= 0.8:
            return 'very similar'
        elif ratio >= 0.6:
            return 'relatively similar'
        elif ratio >= 0.4:
            return 'not very similar'
        else:
            return 'very different'

    return 'undefined'





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

    img_control = img_1.copy()

    # Métricas de controle
    ssim_ctrl = get_ssim(img_1, img_control)[0]
    r_mse_ctrl, r_rmse_ctrl = get_mse_rmse(img_1, img_control)
    hc_ctrl = get_histogram_comparison(img_1, img_control, "histogram_comparison_control", "Histogram Comparison Control")
    fem_ctrl = get_feature_em(img_1, img_control, "feature_em_matched_image_control")[0]
    ncc_ctrl = calculate_ncc(img_1, img_control)[0]
    mi_ctrl = calculate_mi(img_1, img_control)[0]

    # Métricas imagem 1 vs imagem 2
    ssim = get_ssim(img_1, img_2)[0]
    r_mse, r_rmse = get_mse_rmse(img_1, img_2)
    hc = get_histogram_comparison(img_1, img_2, "histogram_comparison", "Histogram Comparison")
    fem = get_feature_em(img_1, img_2, "feature_em_matched_image")[0]
    ncc = calculate_ncc(img_1, img_2)[0]
    mi = calculate_mi(img_1, img_2)[0]

    results = {
        'ssim': [ssim, ssim_ctrl, classify_similarity(ssim, ssim_ctrl, "ssim")],
        'r_mse': [r_mse, r_mse_ctrl, classify_similarity(r_mse, r_mse_ctrl, "r_mse")],
        'r_rmse': [r_rmse, r_rmse_ctrl, classify_similarity(r_rmse, r_rmse_ctrl, "r_rmse")],
        'hc_correlation': [hc[0], hc_ctrl[0], classify_similarity(hc[0], hc_ctrl[0], "hc_correlation")],
        'hc_chi_square': [hc[1], hc_ctrl[1], classify_similarity(hc[1], hc_ctrl[1], "hc_chi_square")],
        'hc_intersection': [hc[2], hc_ctrl[2], classify_similarity(hc[2], hc_ctrl[2], "hc_intersection")],
        'hc_bhattacharyya': [hc[3], hc_ctrl[3], classify_similarity(hc[3], hc_ctrl[3], "hc_bhattacharyya")],
        'fem': [fem, fem_ctrl, classify_similarity(fem, fem_ctrl, "fem")],
        'ncc': [ncc, ncc_ctrl, classify_similarity(ncc, ncc_ctrl, "ncc")],
        'mi': [mi, mi_ctrl, classify_similarity(mi, mi_ctrl, "mi")]
    }


    return results
