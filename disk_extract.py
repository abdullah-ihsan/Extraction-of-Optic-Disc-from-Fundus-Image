import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as mp
import math
from scipy.spatial import distance

def draw_crosshair(image, point, size, color=(0, 255, 0), thickness=1):
    """
    Draw a crosshair on the image at the specified point.

    Args:
        - image: The input image.
        - point: The point (x, y) where the crosshair will be drawn.
        - size: The size of the crosshair.
        - color: The color of the crosshair (BGR format).
        - thickness: The thickness of the crosshair lines.
    """
    x, y = point
    x, y = int(x), int(y)
    half_size = size // 2
    cv.line(image, (x - half_size, y), (x + half_size, y), color, thickness)
    cv.line(image, (x, y - half_size), (x, y + half_size), color, thickness)


def imageMorph(filename):
    img = cv.imread('Fundus image/'+filename)
    # vessel = cv.imread('Blood vessels/'+filename.split('.')[0]+'_seg.png', 0)

    B,G,R = img[:,:,0], img[:,:,1], img[:,:,2]
 
    # cv.imshow('Ginitial', G)
    h, w = G.shape
    if h > 3000:
        kernel = np.ones((75,75),np.uint8)
    elif h > 2400:
        kernel = np.ones((60,60),np.uint8)  
    elif h > 1800:
        kernel = np.ones((45,45),np.uint8)
    elif h > 1200:
        kernel = np.ones((30,30),np.uint8)
    else:
        kernel = np.ones((15,15),np.uint8)
    # kernel2 = np.ones((5,5),np.uint8)

    G = cv.GaussianBlur(G,(9,9),0)
    # G[vessel > 0] = 0
    G = cv.morphologyEx(G, cv.MORPH_CLOSE, kernel)
    # cv.imshow('gg', G)
    G = cv.morphologyEx(G, cv.MORPH_OPEN, kernel)
    G = cv.dilate(G,kernel,iterations = 2)
    # cv.imshow('morph applied',G)
    return G

def ROIdetection(image, real_centres, filename):
    X, Y = real_centres
    _, image_ = cv.threshold(image,150,255,cv.THRESH_BINARY)
    output = cv.connectedComponentsWithStats(image_, 8, cv.CV_32S)
    num_labels = output[0]
    if num_labels == 1:
        m = np.max(image)
        image[(image == m) | (image > (m - 10))] = 255
        _, image = cv.threshold(image,150,255,cv.THRESH_BINARY)
        output = cv.connectedComponentsWithStats(image, 8, cv.CV_32S)
    else:
        image = image_
    # cv.imshow('gg',image)
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]   
    ROI = 1
    if len(labels) > 2:
        diffs = []
        Min = abs(stats[0][2] - stats[0][3])
        for l in range(1,len(stats)):
            temp = abs(stats[l][2] - stats[l][3])
            if temp < Min:
                Min = temp
                ROI = l
    elif len(labels) == 2:
        ROI = 2

    labels = np.uint8(labels * (255 / np.max(labels)))
    labels = cv.cvtColor(labels,cv.COLOR_GRAY2RGB)
    x, y, w, h, area = stats[ROI]
    centre = centroids[ROI]
    
    print('Blood vessels/'+filename+': ' + str(centre) + ' | Error: ' + str(distance.euclidean((int(centre[1]),int(centre[0])),(Y[0],X[0]))))

    # Draw the bounding box 
    og_image = cv.imread('Fundus image/'+filename)
    
    cv.rectangle(labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_crosshair(labels, centre, size=10, color=(0, 0, 255), thickness=1)
    cv.rectangle(og_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_crosshair(og_image, centre, size=10, color=(255, 0, 0), thickness=1)
    draw_crosshair(og_image, (X[0], Y[0]), size=10, color=(255, 255, 0), thickness=1)
    cv.imshow('labels', labels)
    cv.imshow('output', og_image)
    cv.waitKey()