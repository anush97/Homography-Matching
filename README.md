# Edge Detection and Hough Transform Project

This project implements the Hough Transform for line detection on an image. The goal is to find two lines, which can be useful for various applications, such as finding lanes on a road in an image.

## Requirements
To run the code you'll need the following Python packages installed:
- numpy
- skimage
- matplotlib

You can install these packages using pip:
'pip install numpy scikit-image matplotlib'


## Usage

The project consists of two main scripts, utils.py and hough.py. 

### utils.py

This script contains utility functions for the main script hough.py. 

- `create_mask(H, W)` function creates a mask of size (H, W) which is used to extract the region of interest (ROI) from the image. ROI is the region where we expect our lines to be. 

- `create_line(rho, theta, img)` function creates a line in the image space from the given parameters rho and theta in the Hough space. The output is a list of x and y coordinates of the points that lie on the line. 

### hough.py

This is the main script which loads an image, applies the Canny edge detector to find edge points in the image, creates a mask for the ROI and extracts edge points in the ROI, applies the Hough transform, and finally plots the detected lines on the image. 

The script uses skimage to load the image and to detect edges, numpy to perform calculations, and matplotlib to display the results.

To run this script, you can use the command:
'python hough.py'

## Outputs

- "Canny edges": Image with edges detected by the Canny edge detector.
- "Mask": Binary mask of the ROI.
- "Edges within ROI": Image with edges detected within the ROI.
- "Detected Edges": Final result showing two detected lines on the original image.

## Notes

- You can adjust the region of interest by modifying the `create_mask` function.
- Parameters rho and theta in the `create_line` function are in polar coordinates, which represents a line in the Hough space.
- The `hough_transform` function takes an edge image, a range of angles and distances as inputs, and outputs the accumulator array (Hough space) where each cell represents a line in the image space.
- The lines are drawn on the original image in blue and red colors. You can modify the color and line width in the `plt.plot` function calls.

## Conclusion

This is a simple demonstration of edge detection and Hough Transform for line detection in images. You might need to fine-tune some parameters or modify the functions depending on your specific use case.

