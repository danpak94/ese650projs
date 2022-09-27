# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:36:11 2017

@author: Daniel
"""

import numpy as np
import cv2
from numpy import cos,sin,log,exp,pi
import matplotlib.pyplot as plt

# convert meters to pixels for lidar scans
def dist2Pixel(distX,distY,MAP):
    pixelX=np.ceil( (distX-MAP['xmin']) / MAP['res'] ).astype(np.int64)-1
    pixelY=np.ceil( (distY-MAP['ymin']) / MAP['res'] ).astype(np.int64)-1
    return np.vstack((pixelX,pixelY))

def removeFloorScans(lidar_data,HeadAngle,LidarHeight,timeInd):
    minHeight = 0.2
    
    # front to back
    headAngleVec = np.linspace(-HeadAngle[timeInd],HeadAngle[timeInd],1440/2)[0:541]
    
    #-135 to 0 degrees of laser scan
    heightScan1 = LidarHeight + lidar_data[timeInd]['scan'][0,0:541]*sin(np.fliplr([headAngleVec])[0])
    
    #0 to 135 degrees of laser scan
    heightScan2 = LidarHeight + lidar_data[timeInd]['scan'][0,541:1081]*sin(headAngleVec[1:541])
    
#    heightScan = lidar_data[timeInd]['scan']*sin(HeadAngle[timeInd]) 
    indices1 = (heightScan1 >=minHeight)
    indices2 = (heightScan2 >=minHeight)
    
    indices = np.expand_dims(np.concatenate([indices1,indices2]),axis=0)
    indices=np.ones(indices.shape)
    return indices


# get occupied indices in global frame in pixels
def getOccupiedInds(lidar_data,MAP,currentPos,thetaNeck,headAngleValidInd,HeadAngle,timeIdx):
    
    currentX = currentPos[0]
    currentY = currentPos[1]
    currentThetaCOM = currentPos[2]
    
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges=lidar_data[timeIdx]['scan'].T # specify timeIdx ranges=lidar0[timeIdx]['scan'].T
    
    headAngleVec = np.linspace(-HeadAngle[timeIdx],HeadAngle[timeIdx],1440/2)[0:541]
    ranges1 = ranges[0:541,0]*cos(np.fliplr([headAngleVec])[0])
    ranges2 = ranges[541:1081,0]*cos(headAngleVec[1:541])
    ranges = np.expand_dims(np.concatenate([ranges1,ranges2], axis=0),axis=1)
    
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    indValid = np.logical_and(indValid,headAngleValidInd)

    ranges = ranges[indValid]
    angles = angles[indValid]
    
    # occupied physical locations of laser measurements (in meters) in sensor frame
    xs0 = np.array([ranges*np.cos(angles)]);
    ys0 = np.array([ranges*np.sin(angles)]);
    
    lidarFrame = np.vstack((xs0,ys0))
    
    bodyFrame = np.dot( np.array([[cos(thetaNeck[timeIdx]),-sin(thetaNeck[timeIdx])], \
                                     [sin(thetaNeck[timeIdx]),cos(thetaNeck[timeIdx])]]), \
                          lidarFrame )
    
    globalFrame = np.dot( np.array([[cos(currentThetaCOM),-sin(currentThetaCOM)], \
                                     [sin(currentThetaCOM),cos(currentThetaCOM)]]), \
                          bodyFrame )
    
    globalFrame[0,:] = globalFrame[0,:] + currentX
    globalFrame[1,:] = globalFrame[1,:] + currentY
    
    # convert from distance to pixels
    pixelInds = dist2Pixel(globalFrame[0,:],globalFrame[1,:],MAP)
    
    return pixelInds

def getUniqueOrderedPairs(pixelInds):
    # get rid of duplicate ordered pairs of indices
    numpy2tuple = tuple(map(tuple, pixelInds.T))
    output = np.array(list(set(numpy2tuple))).T
    return output

def ryanGetMapCellsFromRay(MAP,occupiedInd,currentCoord):
    
    occupiedInd=occupiedInd.T
    
    MAP['logOdds'][occupiedInd[:,0],occupiedInd[:,1]] += 2*log(9.)
    
    occupiedIndCnt=np.vstack((occupiedInd[:,1],occupiedInd[:,0])).T
    currentCoordPixel=dist2Pixel(currentCoord[:,1],currentCoord[:,0],MAP).T
    occupiedIndCnt=np.append(occupiedIndCnt,currentCoordPixel,axis=0)
    
    mask=np.zeros(MAP['logOdds'].shape)
    cv2.drawContours(image=mask, contours=[occupiedIndCnt], contourIdx=0, \
                     color=log(1/9.)/2, thickness=-1)
    
    MAP['logOdds'] += mask
    MAP['logOdds'][MAP['logOdds']>100] = 100.
    
    MAP['plot'] = 1-1/(1+exp(MAP['logOdds']))
    MAP['map'] = (MAP['plot']>0.5)
    
    return MAP


# INPUT 
# MAP              the map dictionary
# Y(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
# xs,ys           physical x,y,positions you want to evaluate "correlation" (centered around current robot x,y coordinate)
# xs,ys           angle is specified previously in Y(0:2,:)
#
# OUTPUT 
# c               sum of the cell values of all the positions hit by range sensor

def getNextTsParticlePose(prev_pose,lidar_data,idxLidar,timeInd,prev_timeInd):
    
    i_lidar=idxLidar[timeInd]
    prev_i_lidar=idxLidar[prev_timeInd]
    
    xNoise=np.random.normal(0,0.01)
    yNoise=np.random.normal(0,0.01)
#    thetaNoise=np.random.normal(0,0.05*2*pi/40) # 40 because distance is defined -20 to 20 and trying to put comparable noise in angles
#    if abs(lidar_data[i_lidar]['pose'][0,2]-lidar_data[prev_i_lidar]['pose'][0,2])>0.025:
#        print('high noise yaw from pose')
#        thetaNoise=np.random.normal(0,0.45*2*pi/40)
#    elif abs(lidar_data[i_lidar]['rpy'][0,2]-lidar_data[prev_i_lidar]['rpy'][0,2])>0.05:
#        print('high noise from rpy')
#        thetaNoise=np.random.normal(0,0.3+2*abs(lidar_data[i_lidar]['rpy'][0,2]-lidar_data[prev_i_lidar]['rpy'][0,2])*2*pi/40)
#    else:
    thetaNoise=np.random.normal(0,0.05*2*pi/40)
        
    
#    xNoise=0
#    yNoise=0
#    thetaNoise=0
        
#    dxdydtheta_global=lidar_data[i_lidar+1]['pose']-lidar_data[i_lidar]['pose']
    dxdydtheta_global=lidar_data[i_lidar]['pose']-lidar_data[prev_i_lidar]['pose']
    
    yaw_prev=lidar_data[i_lidar]['pose'][0,2]
    dxdy_local=np.dot(np.array([[cos(yaw_prev),sin(yaw_prev)],[-sin(yaw_prev),cos(yaw_prev)]]),dxdydtheta_global[:,0:2].T)
    dtheta_local=dxdydtheta_global[:,2]
    
    yaw_est=lidar_data[i_lidar]['rpy'][0,2]
    dxdy_myGlobal=np.dot(np.array([[cos(yaw_est),-sin(yaw_est)],[sin(yaw_est),cos(yaw_est)]]),dxdy_local)
    dtheta_myGlobal=np.copy(dtheta_local)
    
    xDummy=prev_pose[0]+dxdy_myGlobal[0,:]+xNoise
    yDummy=prev_pose[1]+dxdy_myGlobal[1,:]+yNoise
    thetaCOMdummy=prev_pose[2]+dtheta_myGlobal+thetaNoise
    
    return np.squeeze(np.array([xDummy, yDummy, thetaCOMdummy]))

#def getY(lidar_data,timeInd):
#    angles=np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
#    ranges=lidar_data[timeInd]['scan'].T
#    
#    # take valid indices
#    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
#    ranges = ranges[indValid]
#    angles = angles[indValid]
#    
#    xs0 = np.array([ranges*np.cos(angles)]);
#    ys0 = np.array([ranges*np.sin(angles)]);
#    
#    Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)
#    
#    return Y

def getMapCorrelation(MAP, occupiedInd, currentCoord, xs_size, ys_size):
    
    mapCorrelation = np.sum(MAP['map'][occupiedInd[0,:],occupiedInd[1,:]])
    return mapCorrelation
    
#    nx = MAP['map'].shape[0]
#    ny = MAP['map'].shape[1]
#    
#    nxs = xs_size
#    nys = ys_size
#    
#    xs = np.arange(-int(xs_size/2),int(xs_size/2)+1,1)
#    ys = np.arange(-int(ys_size/2),int(ys_size/2)+1,1)
#    
#    cpr = np.zeros((nxs, nys))
#    for jy in range(0,nys):
#        iy = occupiedInd[1,:] + ys[jy] # 1 x 1076
#        for jx in range(0,nxs):
#            ix = occupiedInd[0,:] + xs[jx] # 1 x 1076
#            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
#			                        np.logical_and((ix >=0), (ix < nx)))
#            cpr[jx,jy] = np.sum(MAP['map'][ix[valid],iy[valid]])
#    
#    return cpr

#def updateParticleWeight():
    

def calc_N_eff(particleWeights):
    output=1/np.sum(particleWeights**2)
    return output

def stratifiedSampling(pose,weights,nParticle):
    newPose=np.zeros(pose.shape)
    newWeights=1/float(nParticle)*np.ones(weights.shape)
    
    cdf=np.cumsum(weights)
    
    u=np.random.uniform(0,1/float(nParticle),1)[0]
    
    j=0
    for k in range(0,nParticle):
        beta=u+k/float(nParticle)
        while cdf[j]<beta:
            j+=1
        
        newPose[k,:]=pose[j,:]
#        print(j)
    return newPose, log(newWeights), cdf
            

def getTimeInds(lidar_data,joint_data):
    tsLidar=np.asarray([l['t'][0,0] for l in lidar_data])
    tsJoint=joint_data['ts']
    
    tMinLidar=np.min(tsLidar)
    tMaxLidar=np.max(tsLidar)
    
    tMinJoint=np.min(tsJoint)
    tMaxJoint=np.max(tsJoint)
    
    maxT=min(tMaxLidar,tMaxJoint)
    minT=max(tMinLidar,tMinJoint)
    
    idxMinLidar=np.where(np.abs(tsLidar-minT)==np.min(np.abs(tsLidar-minT)))[0][0]
    idxMaxLidar=np.where(np.abs(tsLidar-maxT)==np.min(np.abs(tsLidar-maxT)))[0][0]
    idxTotal=idxMaxLidar-idxMinLidar
    
    idxLidar=np.arange(idxMinLidar,idxMaxLidar,1)
    
    timeVec=tsLidar[idxLidar]
    
    idxJoint=np.zeros((1,idxTotal))
    for i in range(0,idxTotal):
        dummy,idx=np.where(np.abs(tsJoint-timeVec[i])==np.min(np.abs(tsJoint-timeVec[i])))
        idxJoint[0,i]=idx[0]-idxMinLidar
    
    idxJoint=np.squeeze(idxJoint).astype(int)
    
    return idxLidar, idxJoint
    
    
def getTimeInds2(lidar_data,joint_data,RGB_data,IR_data):
    
    tsLidar=np.asarray([l['t'][0,0] for l in lidar_data])
    tsJoint=joint_data['ts']
    tsRGB=np.asarray([l['t'][0,0] for l in RGB_data])
    tsIR=np.asarray([l['t'][0,0] for l in IR_data])
    
    tMinRGB=np.min(tsRGB)
    tMaxRGB=np.max(tsRGB)
    
    tMinJoint=np.min(tsJoint)
    tMaxJoint=np.max(tsJoint)
    
    tMinLidar=np.min(tsLidar)
    tMaxLidar=np.max(tsLidar)
    
    tMinIR=np.min(tsIR)
    tMaxIR=np.max(tsIR)
    
    maxT=np.min(np.array([tMaxRGB,tMaxJoint,tMaxLidar,tMaxIR]))
    minT=np.max(np.array([tMinRGB,tMinJoint,tMinLidar,tMinIR]))
    
    idxMinRGB=np.where(np.abs(tsRGB-minT)==np.min(np.abs(tsRGB-minT)))[0][0]
    idxMaxRGB=np.where(np.abs(tsRGB-maxT)==np.min(np.abs(tsRGB-maxT)))[0][0]
    idxTotal=idxMaxRGB-idxMinRGB

    idxRGB = np.arange(idxMinRGB,idxMaxRGB,1)
    
    timeVec=tsRGB[idxRGB]
    
    idxJoint=np.zeros((1,idxTotal))
    idxLidar=np.zeros((1,idxTotal))
    idxIR=np.zeros((1,idxTotal))
    for i in xrange(0,idxTotal):
        idx = np.argmin(np.abs(tsJoint-timeVec[i]))
        idxJoint[0,i]=idx-idxMinRGB
        
        idx = np.argmin(np.abs(tsLidar-timeVec[i]))
        idxLidar[0,i]=idx-idxMinRGB
        
        idx = np.argmin(np.abs(tsIR-timeVec[i]))
        idxIR[0,i]=idx-idxMinRGB
        
    idxJoint=np.squeeze(idxJoint).astype(int)
    idxLidar=np.squeeze(idxLidar).astype(int)
    idxIR=np.squeeze(idxIR).astype(int)
    
    return idxLidar, idxJoint, idxRGB, idxIR

    
    
    
    
    
    
    
    