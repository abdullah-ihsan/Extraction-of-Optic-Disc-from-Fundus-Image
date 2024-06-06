# Retinal Image Segmentation and Objects Analysis
## Overview
Fundus images are digital images of human retinal which are used to diagnose different retinal diseases. This project implements image morphological techniques to extract the optic disc from the fundus image using image processing techniques. 

## Tools and dependencies
- Python 3.x
- OpenCV
- NumPy
- Pandas

## Working and methodology

1. Extracted green channel from the BGR spectrum of the image and applied a blurring filter on it (Gaussian filter)
![image](https://github.com/abdullah-ihsan/Extraction-of-Optic-Disc-from-Fundus-Image/assets/65601738/1f344c46-ba68-4ce1-8d95-3a61f2f22f7d)

2. Closing morphological operation is applied to close off any veins from the image and then opening along with dilation to improve visibility of the optic disc.
![image](https://github.com/abdullah-ihsan/Extraction-of-Optic-Disc-from-Fundus-Image/assets/65601738/931a7f08-acff-40b2-b746-c7b21f6f0bd2)
![image](https://github.com/abdullah-ihsan/Extraction-of-Optic-Disc-from-Fundus-Image/assets/65601738/4c525cf1-2b0b-40f4-be70-550c73067965)

4. The image is thresholded to extract the brighter regions from the background. The centre is then determinied using component labelling and highlighted on the orignal image.
![image](https://github.com/abdullah-ihsan/Extraction-of-Optic-Disc-from-Fundus-Image/assets/65601738/81dd371a-5991-4264-a0df-3d61764bcb2e)
 
## Results
![image](https://github.com/abdullah-ihsan/Extraction-of-Optic-Disc-from-Fundus-Image/assets/65601738/c11293fd-349d-4ed2-aac5-d59870a1b93c)
There is very small difference between the calculated distances and the actual distances of the images. An average error of about **100 pixels** was recorded. The image size varied from 800x800 px to 3000x3000 px, hence the high pixel count of the error.
