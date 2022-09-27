# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:35:52 2017

@author: Daniel
"""

#import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import p3_util as ut
import numpy as np
from numpy import cos,sin,log,exp,pi
import proj3functions as p3f
import time

#%% load data

# list, each index is a time point
lidar0=ld.get_lidar('../Proj3_2017_Train/data/train_lidar0')

# numpy array, second dim index is time point
joint0=ld.get_joint('../Proj3_2017_Train/data/train_joint0')

# Joint more frequent sampling than lidar
idxLidar,idxJoint=p3f.getTimeInds(lidar0,joint0)

lidar_data=lidar0
joint_data=joint0

#%% predict robot position (SE2) based on odometry and imu yaw

start_time = time.time()

MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
MAP['emptyMap'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8)
MAP['plot'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64)
MAP['logOdds'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64)

COMheight = 0.93
HeadHeight = COMheight + 0.33
IMUHeight = COMheight + 0.16
LidarHeight = HeadHeight + 0.15
KinectHeight = HeadHeight + 0.07

NeckAngle = joint0['head_angles'][0,idxJoint] # yaw
HeadAngle = joint0['head_angles'][1,idxJoint] # pitch

#%

# to make map correlation 9x9 for each particle
x_range = np.arange(-0.2,0.2+MAP['res'],MAP['res']).astype(float)
y_range = np.arange(-0.2,0.2+MAP['res'],MAP['res']).astype(float)

nParticle = 100

predictionParticlePose = np.zeros((idxLidar.size,nParticle,3)) # x, y, theta
xs_size=1
ys_size=1
mapCorrelation = np.zeros((nParticle,xs_size,ys_size))
#xs = (np.arange(-int(xs_size/2),int(xs_size/2)+1,1)/float(MAP['map'].shape[0]))
#ys = (np.arange(-int(ys_size/2),int(ys_size/2)+1,1)/float(MAP['map'].shape[1]))
xs = (np.arange(-72,72+18,18)/float(MAP['map'].shape[0]))
ys = (np.arange(-72,72+18,18)/float(MAP['map'].shape[1]))
#mapCorrelation = np.zeros(nParticle)
maxMapCorrelationPos_store = [(5,5)]*nParticle
w = log(1/float(nParticle)*np.ones(nParticle)) # log particle weight

currentPos = np.array([0.,0.,0.])
currentPos_store = np.zeros((idxLidar.size,3))

#for timeInd in range(0,len(lidar0)-1):
startInd=0
endInd=idxLidar.size-1
#endInd=
increment=10
for timeInd in xrange(startInd,endInd-increment,increment):
    print(timeInd)
    
#    currentPos = np.array([x[timeInd],y[timeInd],thetaCOM[timeInd]])
#    currentPos = predictionParticlePose[timeInd,np.argmax(w),:]
    
    if np.argwhere(w==np.max(w)).size == 1:
        xAdd = xs[maxMapCorrelationPos_store[np.argmax(w)][0]]
        yAdd = ys[maxMapCorrelationPos_store[np.argmax(w)][1]]
#        xAdd=0.0
#        yAdd=0.0
    else:
        print('multiple argmaxW')
        xAdd = 0.0
        yAdd = 0.0
#        
    currentPos[0] = np.copy(predictionParticlePose[timeInd,np.argmax(w),0]) + xAdd
    currentPos[1] = np.copy(predictionParticlePose[timeInd,np.argmax(w),1]) + yAdd
    currentPos[2] = np.copy(predictionParticlePose[timeInd,np.argmax(w),2])

    currentPos_store[timeInd,:] = np.copy(currentPos)
    
    ## Choose one best particle with best map correlation and use that to update map

    # passed in as argument of getOccupiedInds
    headAngleValidInd = p3f.removeFloorScans(lidar0,HeadAngle,LidarHeight,timeInd).T
    
    # in pixels, already takes into account robot pose given by best particle
    occupiedInd = p3f.getOccupiedInds(lidar0,MAP,currentPos,NeckAngle,headAngleValidInd,HeadAngle,timeInd)
    indGood = np.logical_and(np.logical_and(np.logical_and((occupiedInd[0,:] > 0), (occupiedInd[1,:] > 0)), \
                                        (occupiedInd[0,:] < MAP['sizex'])), (occupiedInd[1,:] < MAP['sizey']))
#    indValid = np.logical_and((lidar0[timeInd]['scan'] < 30),(lidar0[timeInd]['scan']> 0.1))
#    indGood = np.logical_and(indGood, indValid)
    occupiedInd = occupiedInd[:,indGood]
    # required for closing the contour for ryanGetMapcellsFromRay (in meters, not pixels)
    currentCoord = np.array([[currentPos[0],currentPos[1]]])
    
    # Update both MAP['logOdds'] and MAP['map']
    MAP = p3f.ryanGetMapCellsFromRay(MAP,occupiedInd,currentCoord)
    
#    print(1/np.sum(exp(w)**2))
    if 1/np.sum(exp(w)**2) < 2:
        print('resampling')
        predictionParticlePose[timeInd,:,:],w,cdf=p3f.stratifiedSampling(predictionParticlePose[timeInd,:,:],\
                                                                      exp(w),nParticle)
    
    ## continue trajectory prediction with each particle and calculate map correlation for each particle
    ## only resample when below effective nunmber of particles (N_eff)

    for particleInd in range(0,nParticle):
        predictionParticlePose[timeInd+increment,particleInd,:] = \
                              p3f.getNextTsParticlePose(predictionParticlePose[timeInd,particleInd,:],\
                                                        lidar0,idxLidar,timeInd+increment,timeInd)
                              
        headAngleValidInd = p3f.removeFloorScans(lidar0,HeadAngle,LidarHeight,timeInd+increment).T
        occupiedInd = p3f.getOccupiedInds(lidar0,MAP,predictionParticlePose[timeInd+increment,particleInd,:],\
                                          NeckAngle,headAngleValidInd,HeadAngle,timeInd+increment)
        indGood = np.logical_and(np.logical_and(np.logical_and((occupiedInd[0,:] > 0), (occupiedInd[1,:] > 0)), \
                                                (occupiedInd[0,:] < MAP['sizex'])), (occupiedInd[1,:] < MAP['sizey']))
#        indValid = np.logical_and((lidar0[timeInd]['scan'] < 30),(lidar0[timeInd]['scan']> 0.1))
#        indGood = np.logical_and(indGood,indValid)
        occupiedInd = occupiedInd[:,indGood]
#        uniqueOccupiedInd = p3f.getUniqueOrderedPairs(occupiedInd)
        currentCoord = np.array([[predictionParticlePose[timeInd+increment,particleInd,0],predictionParticlePose[timeInd+increment,particleInd,1]]])
        mapCorrelation[particleInd,:,:] = p3f.getMapCorrelation(MAP,occupiedInd,currentCoord,xs_size,ys_size)
        
#        w[particleInd] = w[particleInd] + mapCorrelation[particleInd,:,:]
        
        maxMapCorrelationPos_store[particleInd] = np.unravel_index(mapCorrelation[particleInd,:,:].argmax(), mapCorrelation[particleInd,:,:].shape) 
        w[particleInd] = w[particleInd] + np.max(mapCorrelation[particleInd,:,:])
    
    w = w-np.max(w)-log(np.sum(exp(w-np.max(w))))
    
    if timeInd%1000==0:
        plt.figure(4)
        plt.imshow(MAP['map'],cmap='gray_r')
        plt.pause(0.001)
    
#%%
MAP['emptyMap'] = (MAP['logOdds']<-30)*0.01
MAP['notExplored'] = np.logical_not(np.logical_or(MAP['logOdds']<-20,MAP['logOdds']>10))*0.5
MAP['combined'] = MAP['map']+MAP['emptyMap']+MAP['notExplored']

#%%

yaw_store = np.zeros(len(lidar0))
for i in xrange(0,len(lidar0)):
    yaw_store[i] = lidar0[i]['rpy'][0,2]

currentPos_store2 = np.dot(np.array([[cos(pi/2),sin(pi/2)],[-sin(pi/2),cos(pi/2)]]),currentPos_store[:,0:2].T)

fig,ax=plt.subplots()
#im=plt.imshow(MAP['map'],cmap='gray')
#im=plt.imshow(MAP['emptyMap'],cmap='gray')
#im=plt.imshow(MAP['notExplored'],cmap='gray')
im=plt.imshow(MAP['combined'],cmap='gray')
fig.colorbar(im)

indices=np.arange(startInd,endInd,increment)
pixelXY=p3f.dist2Pixel(currentPos_store2[0,indices],-currentPos_store2[1,indices],MAP)
plt.plot(pixelXY[0,0:endInd-startInd],pixelXY[1,0:endInd-startInd],'r')