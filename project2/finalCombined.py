# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 01:56:51 2017

@author: Daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 11:23:40 2017

@author: Daniel
"""

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from eulerangles import quat2euler, mat2euler
import proj2functions as p2f
import pdb

fileIdx=1 # options: 1,2,8,9
imageIncrement=25 # every however many indices
filename1='cam/cam'+str(fileIdx)
filename2='imu/imuRaw'+str(fileIdx)
filename3='vicon/viconRot'+str(fileIdx)

x1=io.loadmat(filename1+'.mat')
x2=io.loadmat(filename2+'.mat')
x3=io.loadmat(filename3+'.mat')

maxT=np.min(np.array([np.max(x1['ts']),np.max(x2['ts']),np.max(x3['ts'])]))
minT=np.max(np.array([np.min(x1['ts']),np.min(x2['ts']),np.min(x3['ts'])]))

dummy,idxMin=np.where(np.abs(x2['ts']-minT)==np.min(np.abs(x2['ts']-minT)))
dummy,idxMax=np.where(np.abs(x2['ts']-maxT)==np.min(np.abs(x2['ts']-maxT)))
dummy,idxMinVicon=np.where(np.abs(x3['ts']-minT)==np.min(np.abs(x3['ts']-minT)))

idxMax=idxMax[0]
idxMin=idxMin[0]
idxMinVicon=idxMinVicon[0]

idxTotal=idxMax-idxMin
idxMaxVicon=idxMinVicon+idxTotal

# IMU Accelerometer data
rawAccel=x2['vals'][0:3,idxMin:idxMax] # bits
valueAccel=p2f.raw2valueAccel(rawAccel) # g
valueAccel[0:2,:]=-valueAccel[0:2,:]

# IMU Gyro data
rawGyro=x2['vals'][3:6,idxMin:idxMax] # bits
if filename2 == "imu/imuRaw4":
    valueGyro=p2f.raw2valueGyroIMU4(rawGyro)
else:
    valueGyro=np.copy(p2f.raw2valueGyro(rawGyro)) # rad/s
dt=np.diff(x2['ts'][0,idxMin:idxMax+1])

# calculating quaternion of control input
qu=np.zeros([4,idxTotal])
quTest=np.zeros([4,idxTotal])
for i in range(0,idxTotal):
    gyroEst=np.copy(valueGyro[:,i]*dt[i])
    qu[:,i]=p2f.rVec2quat(np.array([gyroEst[1],gyroEst[2],gyroEst[0]]))

# Just control input, no sigma points: q_{t+1}=q_t*q_\Delta
qt=np.zeros([4,idxTotal])
qt[:,0]=np.array([1,0,0,0])
z1=np.zeros([1,idxTotal])
y1=np.zeros([1,idxTotal])
x1=np.zeros([1,idxTotal])
for i in range(0,idxTotal-1):
    qt[:,i+1]=p2f.quatMult(qt[:,i],qu[:,i])
    [z1[0,i],y1[0,i],x1[0,i]]=quat2euler(qt[:,i])

# Vicon euler angle and gravity vector calculation
z2=np.zeros([1,idxTotal])
y2=np.zeros([1,idxTotal])
x2=np.zeros([1,idxTotal])
viconAccel=np.zeros([3,idxTotal])
for i in range(0,idxTotal):
    [z2[0,i],y2[0,i],x2[0,i]]=mat2euler(x3['rots'][:,:,i+idxMinVicon])
    viconAccel[:,i]=np.squeeze(np.dot(np.transpose(x3['rots'][:,:,i+idxMinVicon]),np.array([[0],[0],[1]])))

# Correcting valueAccel bias using Vicon ground truth
if filename2 == "imu/imuRaw4":
    endIdx=valueAccel.shape[1]
    offset1=np.mean(valueAccel[0,endIdx-50:endIdx]-viconAccel[0,endIdx-50:endIdx])
    offset2=np.mean(valueAccel[1,endIdx-50:endIdx]-viconAccel[1,endIdx-50:endIdx])
    offset3=np.mean(valueAccel[2,endIdx-50:endIdx]-viconAccel[2,endIdx-50:endIdx])
else:
    offset1=np.mean(valueAccel[0,0:50]-viconAccel[0,0:50])
    offset2=np.mean(valueAccel[1,0:50]-viconAccel[1,0:50])
    offset3=np.mean(valueAccel[2,0:50]-viconAccel[2,0:50])
valueAccel[0,:]=valueAccel[0,:]-offset1
valueAccel[1,:]=valueAccel[1,:]-offset2
valueAccel[2,:]=valueAccel[2,:]-offset3

# timeVec for plotting
timeVec=x3['ts'][0,idxMinVicon:idxMaxVicon]-x3['ts'][0,idxMinVicon]

# sigmaPts and vector initialization
n=3
sigmaPtsQuat=np.zeros([4,2*n+1])
preMultiplied=np.zeros([4,2*n+1])
sigmaPtsControlInput=np.zeros([4,2*n+1]) # transformed sigma points in quaternions (q_old*q_sigma*q_delta)
measureSpaceVec=np.zeros([3,2*n+1]) # 3D vectors in measurement space
predictedMeanStore=np.zeros([4,idxTotal])
updatedMeanStore=np.zeros([4,idxTotal])
gravityVec=np.array([0,0,0,1])

priorCov=0.0001*np.eye(3)
processNoise=0.001*np.eye(3)
measurementNoise=0.01*np.eye(3)

#Q=0.0000107*eye(3) # Nikolay's numbers
#R=0.000047*eye(3) # Nikolay's numbers

updatedMean=np.array([1,0,0,0])
updatedCov=np.copy(priorCov)

# range(0,idxTotal)
for j in range(0,idxTotal):
    print(j)
    
    # Calculating sigma points from updatedCov and processNoise    
    sigmaPts=p2f.createSigmaPts(updatedCov+processNoise)
    
    # Adding zero mean as one of the sigma points
    sigmaPts=np.concatenate((sigmaPts,np.zeros((3,1))),axis=1)
    
    # Converting sigma points to quaternions and applying them + control to updatedMean
    for i in range(0,2*n+1):
        sigmaPtsQuat[:,i]=p2f.rVec2quat(sigmaPts[:,i])
        preMultiplied[:,i]=p2f.quatMult(updatedMean,sigmaPtsQuat[:,i])
        sigmaPtsControlInput[:,i]=p2f.quatMult(preMultiplied[:,i],qu[:,j])
    
     # Compute predictedMean for use in update step and processCov for calculating sigma points for next iteration
    predictedMean,e_v_i=p2f.quatMean(np.transpose(sigmaPtsControlInput),updatedMean)
    predictedMeanStore[:,j]=np.copy(predictedMean)
#    updatedMean=np.copy(predictedMean)

#    processCov=p2f.quat2rVecCov(np.transpose(sigmaPtsControlInput),predictedMean)
    processCovDummy=np.zeros([3,3,2*n+1])
    for i in range(0,2*n):
        processCovDummy[:,:,i]=np.outer(e_v_i[:,i],e_v_i[:,i])
    processCov=np.sum(processCovDummy[:,:,0:6],axis=2)/(4*n)+processCovDummy[:,:,6]/(2*n)
#    updatedCov=np.copy(processCov)

    #==============================================================================
    # Add update step
    # Mean of 3D vectors in measurement space and calculating innovation step
    for i in range(0,2*n+1):
        dummy=p2f.quatMult(p2f.quatInverse(sigmaPtsControlInput[:,i]),gravityVec)
        measureSpaceVec[:,i]=p2f.quatMult(dummy,sigmaPtsControlInput[:,i])[1:4]
    
    measurementEst=np.sum(measureSpaceVec[:,0:6],axis=1)/(4*n)+measureSpaceVec[:,6]/(2*n)
    innovation=valueAccel[:,j]-measurementEst
    
    innovationCovDummy=np.zeros([3,3,2*n+1])
    for i in range(0,2*n+1):
        innovationCovDummy[:,:,i]=np.outer(measureSpaceVec[:,i]-measurementEst,measureSpaceVec[:,i]-measurementEst)
    innovationCov=np.sum(innovationCovDummy[:,:,0:6],axis=2)/(4*n)+innovationCovDummy[:,:,6]/(2*n)+measurementNoise
    
    crossCorrelationMatDummy=np.zeros([3,3,2*n+1])
    for i in range(0,2*n+1):
        crossCorrelationMatDummy[:,:,i]=np.outer(e_v_i[:,i],measureSpaceVec[:,i]-measurementEst)
    crossCorrelationMat=np.sum(crossCorrelationMatDummy[:,:,0:6],axis=2)/(4*n)+crossCorrelationMatDummy[:,:,6]/(2*n)
    
    kalmanGain=np.dot(crossCorrelationMat,np.linalg.inv(innovationCov))
    
    dummy=p2f.rVec2quat(np.dot(kalmanGain,innovation))
    updatedMean=p2f.quatMult(dummy,predictedMean)
    
    updatedMeanStore[:,j]=np.copy(updatedMean)
    
    dummy=np.dot(kalmanGain,innovationCov)
    updatedCov=processCov-np.dot(dummy,np.transpose(kalmanGain))
    
z3=np.zeros([1,idxTotal])
y3=np.zeros([1,idxTotal])
x3=np.zeros([1,idxTotal])
for i in range(0,idxTotal-1):
    [z3[0,i],y3[0,i],x3[0,i]]=quat2euler(predictedMeanStore[:,i])
    
z4=np.zeros([1,idxTotal])
y4=np.zeros([1,idxTotal])
x4=np.zeros([1,idxTotal])
for i in range(0,idxTotal-1):
    [z4[0,i],y4[0,i],x4[0,i]]=quat2euler(updatedMeanStore[:,i])

#==============================================================================
# Plotting yaw, pitch, roll of gyro estimate and vicon side by side
plt.figure(1)

plt.subplot(311)
plt.plot(timeVec,z2[0,0:idxTotal],label='Vicon')
plt.plot(timeVec,z1[0,0:idxTotal],label='Gyro Estimate')
#plt.plot(timeVec,z3[0,0:idxTotal],label='Process model')
plt.plot(timeVec,z4[0,0:idxTotal],label='UKF model')
plt.tick_params(labelbottom='off')
plt.legend(prop={'size':10})
plt.title('Yaw')
plt.ylabel('Angle (rad)')

plt.subplot(312)
plt.plot(timeVec,y2[0,0:idxTotal],label='Vicon')
plt.plot(timeVec,y1[0,0:idxTotal],label='Gyro Estimate')
#plt.plot(timeVec,y3[0,0:idxTotal],label='Process model')
plt.plot(timeVec,y4[0,0:idxTotal],label='UKF model')
plt.tick_params(labelbottom='off')
plt.legend(prop={'size':10})
plt.title('Pitch')
plt.ylabel('Angle (rad)')

plt.subplot(313)
plt.plot(timeVec,x2[0,0:idxTotal],label='Vicon')
plt.plot(timeVec,x1[0,0:idxTotal],label='Gyro Estimate')
#plt.plot(timeVec,x3[0,0:idxTotal],label='Process model')
plt.plot(timeVec,x4[0,0:idxTotal],label='UKF model')
plt.legend(prop={'size':10})
plt.title('Roll')
plt.ylabel('Angle (rad)')

plt.xlabel('Time (seconds)')

plt.show()
#==============================================================================

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

x1=io.loadmat(filename1+'.mat') # image data
x2=io.loadmat(filename2+'.mat') # IMU data

dummy,idxMin2=np.where(np.abs(x1['ts']-minT)==np.min(np.abs(x1['ts']-minT)))
dummy,idxMax2=np.where(np.abs(x1['ts']-maxT)==np.min(np.abs(x1['ts']-maxT)))

idxMax2=idxMax2[0]
idxMin2=idxMin2[0]
#print(idxMin,idxMax)

timeVec=x1['ts'][0,idxMin2:idxMax2]
idxTotal2=idxMax2-idxMin2

idxRot=np.zeros((1,idxTotal2))
for i in range(0,idxTotal2):
    dummy,dummy2=np.where(np.abs(x2['ts']-timeVec[i])==np.min(np.abs(x2['ts']-timeVec[i])))
    idxRot[0,i]=dummy2[0]-idxMin

idxRot=np.squeeze(idxRot).astype(int)

R_quat=updatedMeanStore[:,idxRot]
R=np.zeros((3,3,idxTotal2))
for i in range(0,idxTotal2):
    R[:,:,i]=p2f.quat2rmat(R_quat[:,i])

azimuth=flip(np.linspace(-30*pi/180,30*pi/180,num=320))
elevation=flip(np.linspace(-22.5*pi/180,22.5*pi/180,num=240))

sphericalCoord1=np.zeros([240,320,3])
for i in range(0,240):
    for j in range(0,320):
        sphericalCoord1[i,j,:]=np.array([azimuth[j],elevation[i],1])
        
cartesianCoord1=sph2cart(sphericalCoord1)

image=np.zeros([int(pi/(pi/4/240)),int(2*pi/(pi/3/320)),4]).astype(np.uint8)

if fileIdx==1:
    rangeChoice=range(300,idxTotal2-400,imageIncrement)
elif fileIdx==2:
    rangeChoice=range(300,idxTotal2-400,imageIncrement)
elif fileIdx==8:
    rangeChoice=range(200,idxTotal2-200,imageIncrement)
elif fileIdx==9:
    rangeChoice=range(200,idxTotal2-200,imageIncrement)

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
