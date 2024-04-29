import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as mp
import math
import os
from scipy.spatial import distance
import disk_extract

# Load the CSV file into a DataFrame
df = pd.read_csv('centres.csv')


Dir = 'Fundus image/'
# files = os.listdir(Dir)
# for f in files:
# Filter the DataFrame to get rows corresponding to the image name
f = '21_training.tif'
image_data = df[df['image'] == f]
# Extract x and y coordinates from the filtered DataFrame
x = image_data['x'].values
y = image_data['y'].values
morphed = disk_extract.imageMorph(f)
# cv.imshow('m',morphed)
# cv.waitKey()
disk_extract.ROIdetection(morphed, (x,y), f)
# break
