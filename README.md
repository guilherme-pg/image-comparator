# Image Comparator

What it is?</br>
The Image Comparator serves to do exactly what the name says, compare images.</br>
The user selects two images and clicks the 'compare images' button.</br>
Then he is redirected to a page with the metrics and statistics that evaluate the similarity or not between the images.

<img src="./static/images/land_page.png" alt="Land Page" style="height: 400px;
"/></br>

## **How it works?**
Install the requirements.txt.</br>
Preferably in a virtual environment, and in this case I activated it.
After that, using the prompt in the folder where the app is, type:
> python app.py

Copy and paste the http link into your browser and enjoy.


## **Structural Similarity Index Measure (SSIM)**
The SSIM (Structural Similarity Index) score is a metric used to measure the similarity between two images, by quantifying the structural similarity between the pixel intensities of the images, taking into account the perceived changes in structural information, luminance and contrast.</br>
The SSIM score ranges between -1 and 1, with 1 indicating a perfect similarity between the images.</br> 
Higher SSIM scores indicate greater similarity, while lower scores indicate more dissimilarity.</br>
The SSIM score is calculated based on differences in luminance, contrast and structure by comparing the local neighborhood of pixels in the reference image with corresponding pixels in the distorted image.



## **Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)**
MSE (Mean Squared Error) is calculated as the average of the squared differences between the predicted values and the actual values, being obtained through the sum of the squared errors and divided by the total number of data points. </br>
The result is an error value that represents the mean squared difference between the predicted and actual values.</br>
The RMSE (Root Mean Squared Error) is the square root of the MSE.</br>
It is used to interpret the error metric on the same scale as the original data. </br>
Taking the square root, the RMSE provides a measure of the average magnitude of errors in the same unit as the target variable.


## **Histogram Comparison**
<img src="./static/images/histogram_comparison.jpg" alt="Histogram Comparison" style="height: 400px;
"/></br>
Histogram-based methods compare the distribution of pixel intensities in two images.</br>
The result of histogram comparison is a value that indicates the similarity or dissimilarity.</br>
Common techniques when using the histogram:</br>
**Correlation**: Measures the correlation between two histograms by calculating the similarity between the shapes of the 
histograms, which varies between -1 and 1.
Where the value 1 represents a perfect match, 0 represents no correlation, and -1 represents a perfect inverse match.</br>
**Chi-Square**: Calculates the chi-square distance between two histograms by measuring the difference in distribution 
and ranging from 0 to infinity, where higher values indicate greater dissimilarity.</br>
**Intersection**: Calculates the intersection between two histograms, measuring the overlapping area between the 
histograms and varies from 0 to the minimum sum of histogram values, where the higher the greater the similarity.</br>
**Bhattacharyya**: It measures the statistical similarity between distributions and ranges from 0 to 1, so that 0 
represents a perfect match and 1 represents maximum dissimilarity.


## **Feature Extraction and Matching**
<img src="/static/images/feature_em_matched_image.jpg" alt="Feature Extraction and Matching" style="height: 400px;"/></br>
This approach involves extracting features from images (e.g., keypoints, descriptors) and then matching them between 
the two images.</br>
Popular feature extraction algorithms include Scale-Invariant Feature Transform (SIFT), Speeded-Up Robust Features 
(SURF), and Oriented FAST and Rotated BRIEF (ORB).</br>
The similarity score is based on the number of matches, in this case it is the match count and here SIFT was used 
due to its high accuracy.


## **Normalized Cross-Correlation (NCC)**
The NCC score provides a measure of the linear relationship between the pixel intensities of the two images, 
indicating how well the pixel intensities in one image can be predicted from the intensities in the other image.</br>
The NCC score is a value that quantifies the degree of correlation or resemblance between the images, but  it does 
not consider spatial information or the arrangement of pixels.</br>
The resulting Normalized Cross-Correlation value represents the similarity or dissimilarity between the two images, 
with values ranging from -1 to 1.</br>
A value of 1 indicates a perfect match, 0 indicates no correlation, and -1 indicates a perfectanti-correlation.


## **Mutual Information (MI)**
In the context of comparing two images using Mutual Information (MI), the score obtained represents the amount of 
mutual information or shared information between the two images.</br>
Mutual Information is a measure of statistical dependence that quantifies how much knowing the pixel intensities in 
one image reduces the uncertainty about the pixel intensities in the other image.</br>
the obtained MI score can be interpreted as the level of similarity or similarity between the two images based on 
their pixel intensities, where a higher MI score indicates a higher level of shared information and therefore 
suggests a greater degree of similarity between the images.