# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:22:28 2017

@author: Daniel
"""

import math
import numpy as np
from random import random

def gyroEstimate(initOri, gyroValue, dt):
    # Estimate orientation purely based on gyro data
    finalEstimate=initOri+np.dot(gyroValue,dt)
    return finalEstimate
    
def raw2valueAccel(raw):
    # input: 3xn vector of Ax, Ay, Az in bits
    bias=1.5*1023/3.3 # bits
    sensitivity=300 # mV/g
    scale_factor=float(3300)/float(1023)/float(sensitivity)
    value=(raw-bias)*scale_factor # g 
    return value
    
def raw2valueGyro(raw):
    # input: 3xn vector of Wz,Wx,Wy in bits
    bias=np.array([[np.mean(raw[0,0:50])],[np.mean(raw[1,0:50])],[np.mean(raw[2,0:50])]]) # bits
    sensitivity=3.33 # mV/degree/s for x4 gyro
    scale_factor=float(3300)/float(1023)/float(sensitivity)
    value=(raw-bias)*scale_factor*math.pi/180 # rad/s
    return value

def raw2valueGyroIMU4(raw):
    endIdx=raw.shape[1]
    bias=np.array([[np.mean(raw[0,endIdx-50:endIdx])],[np.mean(raw[1,endIdx-50:endIdx])],[np.mean(raw[2,endIdx-50:endIdx])]]) # bits
    sensitivity=3.33 # mV/degree/s for x4 gyro
    scale_factor=float(3300)/float(1023)/float(sensitivity)
    value=(raw-bias)*scale_factor*math.pi/180 # rad/s
    return value

def hatMap(w):
    #converting 1x3 vector to so3 skew-symmetric space
    output=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    return output

def quatNorm(q):
    qs=q[0]
    qv=q[1:4]
    qNorm=qs**2+np.dot(qv,qv)
    return qNorm

def quatInverse(q):
    qs=q[0]
    qv=q[1:4]
    output=np.append(qs,-qv)/quatNorm(q)
    return output

def quatMult(q,p):
#    w1, x1, y1, z1 = q
#    w2, x2, y2, z2 = p
#    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
#    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
#    return np.array([w, x, y, z])
    qs=q[0]
    qv=q[1:4]
    ps=p[0]
    pv=p[1:4]
    qsNew=qs*ps-np.dot(qv,pv)
    qvNew=qs*pv+ps*qv+np.cross(qv,pv)
    qNew=np.append(qsNew,qvNew)
    return qNew

def quatLog(q):
    qs=q[0]
    qv=q[1:4]
    qsNew=math.log(math.sqrt(quatNorm(q)))
    if np.linalg.norm(qv)!=0:
        qvNew=qv/np.linalg.norm(qv)*math.acos(qs/math.sqrt(quatNorm(q)))
    else:
        qvNew=np.array([0,0,0])
    qNew=np.append(qsNew,qvNew)
    return qNew

def quatExp(q):
    qs=q[0]
    qv=q[1:4]
    qsNew=math.exp(qs)*(math.cos(np.linalg.norm(qv)))
    if np.linalg.norm(qv)!=0:
        qvNew=math.exp(qs)*qv/np.linalg.norm(qv)*math.sin(np.linalg.norm(qv))
    else:
        qvNew=np.array([0,0,0])
    qNew=np.append(qsNew,qvNew)
    return qNew

def quat2rVec(q):
    rVec=2*quatLog(q)[1:4]
    return rVec
    
def rVec2quat(rVec):
    a=np.append(0,rVec/2)
    q=quatExp(a)
    return q

def quatMean(qMat,qMeanPrevious):
    nQuat=np.shape(qMat)[0]
#    qGuess=np.array([1,0,0,0])
#    qGuess=np.copy(qMat[0,:])
    qGuess=qMeanPrevious
    w_i=np.ones([nQuat,1])/(2*nQuat)
    e_v_i=np.zeros([3,nQuat])
    
    error=100
    while error > 0.0001:
        for i in range(0,nQuat):
            q_i_e=quatMult(qMat[i,:],quatInverse(qGuess))
            e_v_dummy=quat2rVec(q_i_e)
            e_v_i[:,i]=e_v_dummy
            dummy=np.copy((np.linalg.norm(e_v_dummy)+math.pi)%(2*math.pi))
            if np.linalg.norm(e_v_dummy)!=0:
                e_v_i[:,i]=np.copy((-math.pi+np.copy(dummy))/np.linalg.norm(e_v_dummy)*np.copy(e_v_dummy))
            else:
                e_v_i[:,i]=np.array([0,0,0])
        e_v=np.dot(e_v_i,w_i)
        qGuess=quatMult(rVec2quat(e_v),qGuess)
        error=np.linalg.norm(e_v)
    return qGuess,e_v_i

def quat2rVecCov(qMat,qMean):
    nQuat=np.shape(qMat)[0]
#    e_v_i=np.zeros([3,nQuat])
    covDummy=np.zeros([3,3,nQuat])
    for i in range(0,nQuat):
        q_i_e=quatMult(qMat[i,:],quatInverse(qMean))
        e_v_dummy=quat2rVec(q_i_e)
        dummy=(np.linalg.norm(e_v_dummy)+math.pi)%(2*math.pi)
        if np.linalg.norm(e_v_dummy)!=0:
#            e_v_i[:,i]=(-math.pi+dummy)/np.linalg.norm(e_v_dummy)*np.copy(e_v_dummy)
            e_v_dummy=(-math.pi+dummy)/np.linalg.norm(e_v_dummy)*np.copy(e_v_dummy)
        else:
#            e_v_i[:,i]=np.array([0,0,0])
            e_v_dummy=np.array([0,0,0])
#        covDummy[:,:,i]=np.outer(e_v_i[:,i],e_v_i[:,i])
        covDummy[:,:,i]=np.outer(e_v_dummy,e_v_dummy)
    
    rVecCov=np.sum(covDummy,axis=2)/float(nQuat)
    return rVecCov

def quat2rmat(q):
    qs=q[0]
    qv=q[1:4]
    Eq=np.concatenate((np.expand_dims(-qv,axis=1),qs*np.eye(3)+hatMap(qv)),axis=1)
    Gq=np.concatenate((np.expand_dims(-qv,axis=1),qs*np.eye(3)-hatMap(qv)),axis=1)
    Rq=np.dot(Eq,np.transpose(Gq))
    return Rq

def createSigmaPts(rVecCov):
    # input: 3x3 rotation vector covariance matrix
    L=np.linalg.cholesky(rVecCov)
    n=np.shape(L)[1]
    sigmaPts=np.zeros([3,2*n])
    for i in range(0,n):
        for j in range(0,2):
            sigmaPts[:,2*i+j]=(-1)**j*math.sqrt(2*n)*L[:,i]
#    sigmaPts[:,2*n]=2*quatLog(qMean)[1:4]
    return sigmaPts
    


#def rVecDecompose(rVec):
#    theta=np.linalg.norm(rVec)
#    fixedAxis=rVec/theta
#    return fixedAxis,theta

#def rMat2rVec(rMat):
#    theta=math.acos((np.trace(rMat)-1)/2)
#    logR=theta/(2*math.sin(theta))*(rMat-np.transpose(rMat))
#    rVec=np.array([logR[2,1],logR[0,2],logR[1,0]])
#    return rVec

#def rMat2rVecFixedAxis(rMat):
#    theta=math.acos((np.trace(rMat)-1)/2)
#    print(np.shape(rMat))
#    print(theta)
#    x1=1/(2*math.sin(theta))*(rMat[2,1]-rMat[1,2])
#    x2=1/(2*math.sin(theta))*(rMat[0,2]-rMat[2,0])
#    x3=1/(2*math.sin(theta))*(rMat[1,0]-rMat[0,1])
#    x=np.append(np.append(x1,x2),x3)
#    return x

#def euler2rMat(vec):
#    phi=vec[0]
#    theta=vec[1]
#    psi=vec[2]
#    Rphi=np.array([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]])
#    Rtheta=np.array([[math.cos(theta),0,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]])
#    Rpsi=np.array([[math.cos(psi),-math.sin(psi),0],[math.sin(psi),math.cos(psi),0],[0,0,1]])
#    R=np.dot(np.dot(Rphi,Rtheta),Rpsi)
#    return R