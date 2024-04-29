import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as mp
import math
from scipy.spatial import distance
import os
import disk_extract

df = pd.read_csv('centres.csv')
Dir = 'Fundus image/'
files = os.listdir(Dir)

for f in files:
    img = cv.imread('Fundus image/'+f, 0)
    vessel = cv.imread('Blood vessels/'+f.split('.')[0]+'_seg.png', 0)
    cv.imshow('ds', img)
    cv.imshow('dss', vessel)
    cv.waitKey()
    # vessel_val = []
    # height, width = img.shape
    # for x in range(height):
    #     for y in range(width):
    #         if vessel[x][y] > 0:
    #             vessel_val.append(img[x][y])
    # v = np.array(vessel_val)

    kernel = np.ones((15,15),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imshow('ds', img)
    disk_extract.imageMorph(f)
    cv.waitKey()
    break