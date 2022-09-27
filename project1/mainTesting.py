# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:22:06 2017

@author: Daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 00:29:29 2017

@author: Daniel
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
from math import log,e

folder="C:\Users\Daniel\Google Drive\ESE650\project 1\Training_set"
#folder="D:\Daniel\Google Drive\ESE650\project 1\Training_set"

testInd=np.load('testInd.npy')
barrelRed_mu_k=np.load('barrelRed-mu_k.npy')
barrelRed_sigma_k=np.load('barrelRed-sigma_k.npy')
barrelRed_a_k=np.load('barrelRed-a_k.npy')
notBarrelRed_mu_k=np.load('notBarrelRed-mu_k.npy')
notBarrelRed_sigma_k=np.load('notBarrelRed-sigma_k.npy')
notBarrelRed_a_k=np.load('notBarrelRed-a_k.npy')

for index, fileName in enumerate(os.listdir(folder)):
    if index>-1:
        orgImg=cv2.imread(os.path.join(folder,fileName))
        orgImg=cv2.cvtColor(orgImg,cv2.COLOR_BGR2RGB)       
        img=cv2.cvtColor(orgImg, cv2.COLOR_RGB2HSV)
        
        img=np.reshape(img,(900*1200,3))
        
        # Evaluate likelihoods for each model and get matrix for segmentation
        nClusters=3
        membership=np.empty((nClusters,900*1200))
        membershipNotBarrel=np.empty((nClusters,900*1200))
        for x in range(0,nClusters):
            mu=np.copy(barrelRed_mu_k[x,:])
            sigma=np.copy(barrelRed_sigma_k[x,:,:])

            I=np.copy(img)
            L=np.linalg.cholesky(np.linalg.inv(sigma))
            intermediate=np.dot(I-mu,L)
            g=np.sum(np.square(intermediate),axis=1)
            membership[x,:]=np.exp(-1/2*g)
            
            muNotBarrel=notBarrelRed_mu_k[x,:]
            sigmaNotBarrel=notBarrelRed_sigma_k[x,:]

            InotBarrel=np.copy(img)
            LnotBarrel=np.linalg.cholesky(np.linalg.inv(sigmaNotBarrel))
            intermediateNotBarrel=np.dot(InotBarrel-muNotBarrel,LnotBarrel)
            gNotBarrel=np.sum(np.square(intermediateNotBarrel),axis=1)
            membershipNotBarrel[x,:]=np.exp(-1/2*gNotBarrel)
        
        likelihood=np.dot(np.squeeze(barrelRed_a_k,axis=1),membership)
        likelihoodNotBarrel=np.dot(np.squeeze(notBarrelRed_a_k,axis=1),membershipNotBarrel)
        
        idx=(likelihood>likelihoodNotBarrel)
        idx=np.reshape(idx,(900,1200))
        
        # Morphology transformations
        idxCopy=np.copy(idx)
        idxCopy=img_as_ubyte(idxCopy)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(idxCopy, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        processingDone=opening
        
        # Plotting
        fig, ax = plt.subplots(ncols=3,figsize=(15,5))
        ax[0].imshow(idx,cmap='gray')
        ax[0].set_title('Color segmentation with GMM')
        ax[1].imshow(processingDone,cmap='gray')
        ax[1].set_title('Further processing with openCV')
        ax[2].imshow(orgImg)
        ax[2].set_title('Bounding box on original image')
        label_img = label(processingDone, connectivity=idx.ndim)
        regions = regionprops(label_img)
        print('new group')
        
        # Bounded box and outputting results
        for props in regions:        
            minr, minc, maxr, maxc = props.bbox
            aspectRatio = float((maxr-minr))/float((maxc-minc))
            if 1.15 < aspectRatio < 1.8 and min(maxc-minc,maxr-minr)>25:
                Area=(maxc-minc)*(maxr-minr)
                d=364.98*e**(-0.458*log(Area))
                print('Distance='+str(d))
                print('TopLeftX='+str(minc))
                print('TopLeftY='+str(minr))
                print('BottomRightX='+str(maxc))
                print('BottomRightY='+str(maxr))
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax[2].plot(bx, by, '-g', linewidth=2.5)
        ax[2].axis((0, 1200, 900,0))
        plt.show()

        
print('done')


# Attempt at using opencv for bounding box
#finalIm=np.copy(processingDone)
#contours, hierarchy = cv2.findContours(finalIm,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#cnt = contours[0]
#epsilon = 0.1*cv2.arcLength(cnt,True)
#approx = cv2.approxPolyDP(cnt,epsilon,True)
#print(np.shape(approx))
#if np.shape(approx)[0]==4:
#    cv2.drawContours(finalIm, [approx], -1, (255, 255, 255),3)
#else:
#    rect = cv2.minAreaRect(cnt)
#    box = cv2.cv.BoxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(finalIm,[box],-1,(255,255,255),3)
#    print('boxpoints')
#    print(box)
#
#fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15,10))
#
#axs[0,0].set_title('Original Image')
#axs[0,0].imshow(orgImg)
#
#axs[0,1].set_title('Color segmentation')
#axs[0,1].imshow(idx,cmap='gray')
#
#axs[1,0].set_title('After openCV processing of segmented img')
#axs[1,0].imshow(processingDone,cmap='gray')
#
#axs[1,1].set_title('Bounding box')
#axs[1,1].imshow(finalIm)
