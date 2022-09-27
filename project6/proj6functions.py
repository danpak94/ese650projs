# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:36:58 2017

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth2rgb(DEPTHdata, params):
    # Transforming depth data to rgb camera frame
    
    cx_rgb = params['rgb_param']['cc'][0,0][0,0]
    cy_rgb = params['rgb_param']['cc'][0,0][1,0]
    fx_rgb = params['rgb_param']['f'][0,0][0,0]
    fy_rgb = params['rgb_param']['f'][0,0][1,0]
    
    cx_depth = params['depth_param']['cc'][0,0][0,0]
    cy_depth = params['depth_param']['cc'][0,0][1,0]
    fx_depth = params['depth_param']['f'][0,0][0,0]
    fy_depth = params['depth_param']['f'][0,0][1,0]
    
    # depth_coord = u,v coordinate
    u_depth = np.arange(0, DEPTHdata.shape[1])
    v_depth = np.arange(0, DEPTHdata.shape[0])
    u_depth, v_depth = np.meshgrid(u_depth, v_depth)
    
    # computing x and y coordinates in IR frame
    x_depth = (u_depth - cx_depth) * DEPTHdata / fx_depth
    y_depth = (v_depth - cy_depth) * DEPTHdata / fy_depth
    z_depth = DEPTHdata
    
    x_depth = x_depth.reshape(1,-1)
    y_depth = y_depth.reshape(1,-1)
    z_depth = z_depth.reshape(1,-1)
    
    # 3D point wrt IR frame (x,y,z)
    P_depth = np.concatenate((x_depth, y_depth, z_depth), axis=0)
    
    # 3D point wrt RGB frame (x,y,z)
    P_rgb = np.dot(params['R'], P_depth) + params['T']
    
    x_rgb = P_rgb[0,:]
    y_rgb = P_rgb[1,:]
    z_rgb = P_rgb[2,:]
    
    u_rgb = ( fx_rgb * x_rgb / z_rgb + cx_rgb ).astype(int)
    v_rgb = ( fy_rgb * y_rgb / z_rgb + cy_rgb ).astype(int)
    
    return P_rgb, u_rgb, v_rgb

def getImgColorList(RGBdata, u_rgb, v_rgb):
    # Creating colorList for plotting image with combined RGBD data
    # indexArray is used for testing certain parts of algorithm with a set of points that form a plane

    colorList = []
    indexArray = np.array([]) # storing indices for points in depth camera that actually corresponds to pixels in RGB camera
    for i in xrange(u_rgb.size):
        if (0<=u_rgb[i]<RGBdata.shape[1]) & (0<=v_rgb[i]<RGBdata.shape[0]):
    #        if (1100<u_rgb[i]<1250) & (550<v_rgb[i]<650): # color a patch for test
            if 0: # to keep the if statement above
                colorList.append([255,255,0])
                indexArray = np.append(indexArray, np.int(i))
            else:
                colorList.append(RGBdata[v_rgb[i], u_rgb[i],:])
        else:
            colorList.append([0,255,0])
    
    indexArray = indexArray.astype(int)
    
    return colorList, indexArray

def getNeighborhoodNormal(radius, RGBdata, P_rgb, u_rgb, v_rgb):
     # Defining neighbohood and normal vector of desired points

#    radius = 50 # worked okay (3-4 sec)
#    radius = 100 # worked okay (50 sec)
    
    x_radius = radius # units = mm
    y_radius = radius # units = mm
    z_radius = radius # units = mm
    
    x_rgb = P_rgb[0,:]
    y_rgb = P_rgb[1,:]
    z_rgb = P_rgb[2,:]
    
    normalList = []
    normalPointsList = []
    indexArray = np.array([]) # storing indices for points actually used for normal vector
    for i in xrange(u_rgb.size):
        if i%100 == 0:
            if (0<=u_rgb[i]<RGBdata.shape[1]) & (0<=v_rgb[i]<RGBdata.shape[0]):
                x = x_rgb[i]
                y = y_rgb[i]
                z = z_rgb[i]
                
                x_ind = np.logical_and( (x-x_radius < x_rgb), (x_rgb < x+x_radius) )
                y_ind = np.logical_and( (y-y_radius < y_rgb), (y_rgb < y+y_radius) )
                z_ind = np.logical_and( (z-z_radius < z_rgb), (z_rgb < z+z_radius) )
                
    #            print(np.sum((x_ind & y_ind) & z_ind))
                
                pts4plane = P_rgb[:, (x_ind & y_ind) & z_ind] # neighborhood of points
    
                c = np.mean(pts4plane, axis=1).reshape(-1,1)
                U,s,V = np.linalg.svd((pts4plane-c), full_matrices=True)
#                print((pts4plane-c).shape)
                
                normalList.append(U[:,2].reshape(-1,1))
                normalPointsList.append(P_rgb[:,i].reshape(-1,1))
                
                indexArray = np.append(indexArray, np.int(i))
                
    #        else:
    #            print('not within rgb camera range')
    
    indexArray = indexArray.astype(int)
    
    return normalList, normalPointsList, indexArray

def getClusterColorList(ms, normalsList):
    # Creating colorList for plotting normal vectors with different colors

    randomNums = []
    for k in xrange(len(np.unique(ms.labels_))):
        randomNums.append(np.random.rand(3))

    colorList = []
    
    for i in xrange(len(normalsList)):
        colorList.append((np.array([255,255,255])*randomNums[ms.labels_[i]]).astype(int))
            
    return colorList

def getGroupedColorList(msDist, normalsList, labels):
    # Creating colorList for plotting grouped normal vectors

    randomNums = {}
    for k in xrange(len(np.unique(labels))):
        randomNums[np.unique(labels)[k]] = np.random.rand(3)
    
    colorList = []
    for i in xrange(len(normalsList)):
        colorList.append((np.array([255,255,255])*randomNums[labels[i]]).astype(int))
            
    return colorList

def meanshiftDP(points, h):
    # DP (Daniel Pak)'s implementation of mean shift algorithm with Gaussian kernel
    
    error=1e6
    counter=0
    finalPoints = np.copy(points)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    h1 = ax1.scatter(finalPoints[:,0],finalPoints[:,1],finalPoints[:,2])
    
    while error > 0.000001:
        print(counter)
        
        initialPoints = np.copy(finalPoints)
        
        for k in xrange(points.shape[0]):
            
            eachPoint = np.copy(initialPoints[k,:])
            
#            print('eachPoint')
#            print(eachPoint)
            
            numerator = np.sum( initialPoints * np.exp(-0.5* np.sum( ((eachPoint-points)/h)**2, axis=1) ).reshape(-1,1), axis=0)
            denomenator = np.sum( np.exp(-0.5* np.sum( ((eachPoint-points)/h)**2, axis=1) ), axis=0)
            meanshift = numerator/denomenator - eachPoint
            
#            print('meanshift') 
#            print(meanshift)
            
            finalPoints[k,:] = eachPoint + meanshift
            
#            print(error)
        
        error = np.sum(abs(initialPoints - finalPoints))
        
        print(error)
            
        counter+=1
        
        h1.remove()
        h1 = ax1.scatter(finalPoints[:,0],finalPoints[:,1],finalPoints[:,2], color='r', s=2)
        h1 = ax1.scatter(points[:,0],points[:,1],points[:,2], color='k', s=1)
        
        plt.pause(0.5)
        
    print('done!')

    return finalPoints

##%% SVD to get plane normal
#
#pts4plane = P_rgb[:,indexArray]
#c = np.mean(pts4plane, axis=1).reshape(-1,1)
#U,s,V = np.linalg.svd((pts4plane-c), full_matrices=True)
#
#print(U.shape) # 3x3
#print(s.shape) # 3x1
#print(V.shape) # 1788x1788


##%% grayscale and binary conversion from rgb image
#grayscaleImg = cv2.cvtColor(RGBdata, cv2.COLOR_BGR2GRAY)
#th3 = cv2.adaptiveThreshold(grayscaleImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                            cv2.THRESH_BINARY, 11, 2)
#plt.imshow(th3, cmap='gray')