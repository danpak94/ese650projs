# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:00:25 2017

@author: Daniel
"""

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from math import pi,sin,tan
from rotplot import rotplot

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
for i in range(0,idxTotal):
    dummy,dummy2=np.where(np.abs(x3['ts']-timeVec[i])==np.min(np.abs(x3['ts']-timeVec[i])))
    idxRot[0,i]=dummy2[0]

idxRot=np.squeeze(idxRot).astype(int)
R=x3['rots'][:,:,idxRot]
idxRotTotal=idxRot.shape[0]

azimuth=(np.linspace(-30*pi/180,30*pi/180,num=320))
elevation=(np.linspace(-22.5*pi/180,22.5*pi/180,num=240))

sphericalCoord1=np.zeros([240,320,3])
for i in range(0,240):
    for j in range(0,320):
        sphericalCoord1[i,j,:]=np.array([azimuth[j],elevation[i],1])
        
cartesianCoord1=sph2cart(sphericalCoord1)

#==============================================================================
# Plotting rotation matrices
for i in range(600,1200,25):
    
    fig=plt.figure(1)
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    RTurn1=(x3['rots'][:,:,idxRot[i]])
    rotplot(RTurn1,ax1)
    
    ax2=fig.add_subplot(1,2,2)
    ax2.imshow(x1['cam'][:,:,:,i])
    
    plt.show()
#==============================================================================

#for i in range(300,500,30):
#    plt.imshow(x1['cam'][:,:,:,i])
#    plt.show()