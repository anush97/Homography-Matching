# Template Matching and Homography Project

This project implements image matching and homography to detect a specific template in a target image and then replace the template with a new image. 

## Requirements
To run the code you'll need the following Python packages installed:
- numpy
- matplotlib
- skimage
- OpenCV

You can install these packages using pip:
'pip install numpy matplotlib scikit-image opencv-python'


## Usage

The project consists of three main scripts: utils.py, run.py, and homography.py. 

### utils.py

This script contains utility functions for the main script run.py. 

- `visualize_box(template, target, H)` function visualizes the detected bounding box of the template in the target image.
- `visualize_match(template, target, locs1, locs2, matches)` function visualizes the raw matching results between the template and target image.

### run.py

This is the main script which performs image matching and homography. The script loads images, finds matching pairs between two images, estimates homography, and replaces the template with a new image. 

To run this script, you can use the command:
'python run.py'


### homography.py

This script contains functions related to image matching and homography. 

- `matchPics(I1, I2)` function performs SIFT matching to find candidate match pairs between two images.
- `computeH_ransac(matches, locs1, locs2)` function uses RANSAC to estimate the homography matrix and find inlier matches.
- `compositeH(H, template, img)` function creates a composite image by warping the template image onto the target image using the estimated homography.

## Outputs

- Visualized raw matching result between the template and the target image.
![image](https://github.com/anush97/Homography-Matching/assets/32952140/ace04405-4f7d-4d64-8ef7-7fb9186b2887)
- Visualized matching result after RANSAC.
![image](https://github.com/anush97/Homography-Matching/assets/32952140/c060b518-d2b7-4cd2-86ee-b5ababdf87dc)
- Visualized bounding box in the target image.
![image](https://github.com/anush97/Homography-Matching/assets/32952140/1cb42cff-2705-4bbb-a553-f43f6768e963)
- Final composite image with the template replaced by a new image.
![image](https://github.com/anush97/Homography-Matching/assets/32952140/2658cdfc-321f-4507-b0d1-3eecc7b13311)


## Notes

- The script uses SIFT feature descriptor for image matching. You can replace SIFT with other feature descriptors as needed.
- The script uses the RANSAC algorithm to robustly estimate homography. You can adjust the number of iterations and the distance threshold in the `computeH_ransac` function.
- The `visualize_box` function assumes the homography matrix `H` is in the form of a 3x3 numpy array.
- Make sure the template image and the new image have the same size. If not, you need to resize the new image to the size of the template image.

## Conclusion

This is a demonstration of template matching and homography for image replacement. Depending on your specific use case, you might need to fine-tune some parameters or modify the functions.
