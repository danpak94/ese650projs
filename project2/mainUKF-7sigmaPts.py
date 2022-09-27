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

fileIdx=4
filename1='cam/cam'+str(fileIdx)
filename2='imu/imuRaw'+str(fileIdx)
filename3='vicon/viconRot'+str(fileIdx)

#x1=io.loadmat(filename1+'.mat')
x2=io.loadmat(filename2+'.mat')
x3=io.loadmat(filename3+'.mat')

maxT=np.min(np.array([np.max(x2['ts']),np.max(x3['ts'])]))
minT=np.max(np.array([np.min(x2['ts']),np.min(x3['ts'])]))

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

# Vicon euler angles and gravity vector calculation
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

z4=np.zeros([1,idxTotal])
y4=np.zeros([1,idxTotal])
x4=np.zeros([1,idxTotal])
               
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
plt.figure(2)

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
