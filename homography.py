import numpy as np
import cv2
from skimage.feature import match_descriptors, SIFT
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs
    I2 = rgb2gray(I2)
    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(I1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(I2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.7, cross_check=True)
    locs1 = keypoints1
    locs2 = keypoints2


    ### You can use skimage or OpenCV to perform SIFT matching
    
    ### END YOUR CODE
    
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):
    max_iters = 3500
    thresh = 150
    bestH = None
    best_inliers = []
    num_matches = len(matches)
    
    for i in range(max_iters):
    # randomly sample 4 matches
        sample = np.random.choice(num_matches, 4, replace=False)
        x1 = locs1[matches[sample, 0], 0]
        y1 = locs1[matches[sample, 0], 1]
        x2 = locs2[matches[sample, 1], 0]
        y2 = locs2[matches[sample, 1], 1]
        
         
    # compute homography matrix
    A = np.zeros((8, 9))
    for j in range(4):
      A[j*2, :] = [y1[j], x1[j], 1, 0, 0, 0, -y1[j]*y2[j], -x1[j]*y2[j], -y2[j]]
      A[j*2+1, :] = [0, 0, 0, y1[j], x1[j], 1, -y1[j]*x2[j], -x1[j]*x2[j], -x2[j]]
    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    H = H / H[2,2]     
    # compute inliers
    inliers = []
    for j in range(num_matches):
        x1 = np.append(locs1[matches[j, 0]], 1)
        x2 = np.append(locs2[matches[j, 1]], 1)
        d = np.sqrt(np.sum((x2 - np.dot(H, x1))**2))
        if d < thresh:
            inliers.append(j)
      
    # update best model and inliers
    if len(inliers) > len(best_inliers):
        bestH = H
        best_inliers = inliers


     ### YOUR CODE HERE
     ### You should implement this function using Numpy only
     
     ### END YOUR CODE

    return bestH, inliers

def compositeH(H, template, img):

    # Create a compositie image after warping the template image on top
    # of the image using homography
    

    h, w = template.shape[:2]
    im_h, im_w = img.shape[:2]

    # Create a mask of the size of the template
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Warp mask and template by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H, (im_w, im_h))
    warped_template = cv2.warpPerspective(template, H, (im_w, im_h))

    # Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask > 0] = warped_template[warped_mask > 0]


    
    return composite_img
