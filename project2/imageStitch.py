# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:20:21 2017

@author: Daniel
"""

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def cart2sph(xyz):
    # input: coordinates of each pixel in image matrix
    originalDim1=xyz.shape[0]
    originalDim2=xyz.shape[1]
    originalDim3=xyz.shape[2]
    xyz=np.reshape(xyz,(originalDim1*originalDim2,originalDim3))
    x=xyz[:,0]
    y=xyz[:,1]
    z=xyz[:,2]
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.expand_dims(azimuth,axis=1)
    elevation = np.expand_dims(elevation,axis=1)
    r = np.expand_dims(r,axis=1)
    aer=np.concatenate([azimuth,elevation,r],axis=1)
    aer=np.reshape(aer,(originalDim1,originalDim2,originalDim3))
    return aer

def sph2cart(aer):
    # input: coordinates of each pixel in image matrix
    originalDim1=aer.shape[0]
    originalDim2=aer.shape[1]
    originalDim3=aer.shape[2]
    aer=np.reshape(aer,(originalDim1*originalDim2,originalDim3))
    azimuth=aer[:,0]
    elevation=aer[:,1]
    r=aer[:,2]
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    x = np.expand_dims(x,axis=1)
    y = np.expand_dims(y,axis=1)
    z = np.expand_dims(z,axis=1)
    xyz=np.concatenate([x,y,z],axis=1)
    xyz=np.reshape(xyz,(originalDim1,originalDim2,originalDim3))
    return xyz

def flip(npArray):
    dummy=np.expand_dims(npArray,axis=0)
    output=np.squeeze(np.fliplr(dummy))
    return output

fileIdx=2
imageIncrement=25
filename1='cam/cam'+str(fileIdx)
filename3='vicon/viconRot'+str(fileIdx)

x1=io.loadmat(filename1+'.mat')
x3=io.loadmat(filename3+'.mat')

maxT=np.min(np.array([np.max(x1['ts']),np.max(x3['ts'])]))
minT=np.max(np.array([np.min(x1['ts']),np.min(x3['ts'])]))

dummy,idxMin=np.where(np.abs(x1['ts']-minT)==np.min(np.abs(x1['ts']-minT)))
dummy,idxMax=np.where(np.abs(x1['ts']-maxT)==np.min(np.abs(x1['ts']-maxT)))

idxMax=idxMax[0]
idxMin=idxMin[0]
print(idxMin,idxMax)

timeVec=x1['ts'][0,idxMin:idxMax]
idxTotal=idxMax-idxMin

idxRot=np.zeros([1,idxTotal])
for i in range(idxMin,idxMax):
    dummy,dummy2=np.where(np.abs(x3['ts']-timeVec[i])==np.min(np.abs(x3['ts']-timeVec[i])))
    idxRot[0,i]=dummy2[0]

idxRot=np.squeeze(idxRot).astype(int)
R=x3['rots'][:,:,idxRot]

azimuth=flip(np.linspace(-30*pi/180,30*pi/180,num=320))
elevation=flip(np.linspace(-22.5*pi/180,22.5*pi/180,num=240))

sphericalCoord1=np.zeros([240,320,3])
for i in range(0,240):
    for j in range(0,320):
        sphericalCoord1[i,j,:]=np.array([azimuth[j],elevation[i],1])
        
cartesianCoord1=sph2cart(sphericalCoord1)

image=np.zeros([int(pi/(pi/4/240)),int(2*pi/(pi/3/320)),4]).astype(np.uint8)

if fileIdx==1:
    rangeChoice=range(200,idxTotal-300,imageIncrement)
elif fileIdx==2:
    rangeChoice=range(200,idxTotal-300,imageIncrement)
elif fileIdx==8:
    rangeChoice=range(200,idxTotal-200,imageIncrement)
elif fileIdx==9:
    rangeChoice=range(200,idxTotal-200,imageIncrement)

for k in rangeChoice:
    
    print(k)
    rotatedCartCoord=np.zeros([240,320,3])
    for i in range(0,240):
        for j in range(0,320):
            rotatedCartCoord[i,j,:]=np.dot(R[:,:,k],cartesianCoord1[i,j,:])

    sphericalCoord2=cart2sph(rotatedCartCoord)
    
    xp1=np.array([-pi,pi])
    xp2=np.array([-pi/2,pi/2])
    fp1=np.array([image.shape[1]-1,0])
    fp2=np.array([image.shape[0]-1,0])
    for i in range(0,240):
        for j in range(0,320):
            imgIdx1=int(np.interp(sphericalCoord2[i,j,1],xp2,fp2))
            imgIdx2=int(np.interp(sphericalCoord2[i,j,0],xp1,fp1))
            
            numAccumulation=np.copy(image[i,j,3])
            image[imgIdx1,imgIdx2,0:3]=((image[imgIdx1,imgIdx2,0:3]*float(numAccumulation)+x1['cam'][i,j,:,k])/float(numAccumulation+1)).astype(np.uint8)
            image[imgIdx1,imgIdx2,3]=image[imgIdx1,imgIdx2,3]+1
    
    plt.imshow(image[:,:,0:3])    
    plt.draw()
    plt.pause(0.01)

plt.imshow(image[:,:,0:3])
plt.show()