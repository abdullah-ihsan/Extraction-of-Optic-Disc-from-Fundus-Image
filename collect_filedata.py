import numpy as np
import pandas as pd
import os
import cv2 as cv

m = 'Blood vessels/'
f = 'Fundus image/'
li = os.listdir(m)
fi = os.listdir(f)

somelist = []

for l, o in zip(li, fi):
    i = cv.imread(os.path.join(m, l), 0)
    j = cv.imread(os.path.join(f, o), 0)
    # if i is not None and j is not None:  # Check if images are loaded successfully
    #     if i.shape != j.shape :print(l + ': ' + str(i.shape) + ' | ' + o + ': ' + str(j.shape))
    # else:
    #     print("Failed to load", l, "or", o)
    # if j.shape != (584,565):
        # j_ = cv.resize(j, (584,565))
        # cv.imshow('sdf',j_)
    print(o + ': ' + str(j.shape)) 

    # print(type(j.shape))

    somelist.append(j.shape)
        # cv.waitKey()
some = np.array(somelist)

print(np.unique(some))

