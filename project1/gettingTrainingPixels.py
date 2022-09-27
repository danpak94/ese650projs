# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:59:28 2017

@author: Daniel
"""

import cv2
import os
from roipoly import roipoly
from matplotlib import pyplot as plt
import numpy as np

fileNameToBeSaved='zzz'

folder="C:\Users\Daniel\Google Drive\ESE650\project 1\Training_set"
#folder="D:\Daniel\Google Drive\ESE650\project 1\Training_set"

trainingPixels_stored=[]

for index, fileName in enumerate(os.listdir(folder)):
    img=cv2.imread(os.path.join(folder,fileName))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])

    ROI1=roipoly(roicolor='r')
    
    if ROI1.allxpoints:
        idxMask=(ROI1.getMask(img)==1)
        plt.imshow(idxMask)
        plt.show()
        trainingPixels_stored.append(img[idxMask])
    else:
        trainingPixels_stored.append([])

np.save(fileNameToBeSaved+'.npy',trainingPixels_stored)