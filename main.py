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

df = pd.read_csv('centredata/train_data.csv')
filenames = df.loc[:, 'image'].tolist()
Y = df.loc[:, 'x'].tolist()
X = df.loc[:, 'y'].tolist()

# for i in range(5,6):
for i in range(len(filenames)):
    img = cv.imread('Fundus image/'+filenames[i])
    vessel = cv.imread('Blood vessels/OTSU_'+filenames[i][:-4]+'_seg.png', 0)

    B,G,R = img[:,:,0], img[:,:,1], img[:,:,2]
 
    cv.imshow('Ginitial', G)

    kernel = np.ones((15,15),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    G = cv.GaussianBlur(G,(9,9),0)
    G[vessel > 0] = 0
    G = cv.morphologyEx(G, cv.MORPH_CLOSE, kernel)
    G = cv.morphologyEx(G, cv.MORPH_OPEN, kernel)
    G = cv.dilate(G,kernel,iterations = 2)

    _, G_ = cv.threshold(G,150,255,cv.THRESH_BINARY)

    output = cv.connectedComponentsWithStats(G_, 8, cv.CV_32S)
    num_labels = output[0]
    # print(num_labels)
    if num_labels == 1:
        m = np.max(G)
        G[(G == m) | (G > (m - 10))] = 255
        _, G = cv.threshold(G,150,255,cv.THRESH_BINARY)
        output = cv.connectedComponentsWithStats(G, 8, cv.CV_32S)
    else:
        G = G_
    # cv.imshow('gg',G)
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

    # Convert label matrix to uint8 for visualization
    labels = np.uint8(labels * (255 / np.max(labels)))
    labels = cv.cvtColor(labels,cv.COLOR_GRAY2RGB)
    x, y, w, h, area = stats[ROI]
    centre = centroids[ROI]
    
    print('Blood vessels/'+filenames[i]+': ' + str(centre) + ' | Error: ' + str(distance.euclidean((centre[1],centre[0]),(X[l],Y[l]))))

    # Draw the bounding box 
    cv.rectangle(labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_crosshair(labels, centre, size=10, color=(0, 0, 255), thickness=1)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_crosshair(img, centre, size=10, color=(255, 0, 0), thickness=1)
    
    cv.imshow('labels', labels)
    cv.imshow('output', img)
    cv.waitKey()