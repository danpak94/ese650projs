# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:13:49 2017

@author: Daniel
"""

import numpy as np
from numpy import log
import pdb

def getMovementSpecificVars(movementIdx, K, movementSpecific_z_all_list, movementSpecific_dataList):
    
    # 5 vectors of z's stored for each dataset
    movementSpecific_zList = []
    
    prev_dataIdx = 0
    for k in xrange(0,K):
        dataLength = movementSpecific_dataList[k].shape[0]
        
        movementSpecific_zList.append( movementSpecific_z_all_list[ prev_dataIdx : prev_dataIdx + dataLength ] )
        prev_dataIdx = prev_dataIdx + dataLength
        
    return movementSpecific_zList

def getInitializedVars(N, M):
    # N x N Transition matrix (conditional probabiity from one state to next)
    # T(i,j) = p(s_i | s_j)
    T = np.ones((N,N)) / float(N)
    T = T + np.random.normal(0, 0.1/N, (N,N))
    col_sums = T.sum(axis=0)
    T = T / col_sums[np.newaxis,:]
    
    # M x N Observation/Emission matrix (conditional probability from state to observation)
    # B(i,j) = p(z_i | s_j)
    B = np.ones((M,N)) / float(M)
    B = B + np.random.normal(0, 0.1/M, (M,N))
    col_sums = B.sum(axis=0)
    B = B / col_sums[np.newaxis,:]
    
    # N x 1 Prior (probability of state)
    prior = np.ones(N) / float(N)
    prior = prior + np.random.normal(0, 0.1/N, N)
    col_sums = prior.sum()
    prior = prior / col_sums
    
    return prior, T, B

def BaumWelch(prior, T, B, movementSpecific_dataList, movementSpecific_zList, movementSpecific_z_all, N, M, K):
    # update prior, T, B with Baum-Welch algorithm (EM for HMM)
    
    # Initializ lists (each entry: for each dataset of a specific arm movement)
    
    error=1
    
    while error > 0.01:
        prev_prior = np.copy(prior)
        prev_T = np.copy(T)
        prev_B = np.copy(B)
        
        alphaList=[]
        betaList=[]
        gammaList=[]
        xiList=[]
        for k in xrange(0,K):
#            print('k = ' + str(k))
            
            # choose dataset
            data = movementSpecific_dataList[k]
            z = movementSpecific_zList[k]
            
            # Initialize matrices
            nTimeSteps = data.shape[0]
            alpha = np.zeros((nTimeSteps,N))
            beta = np.zeros((nTimeSteps,N))
            gamma = np.zeros((nTimeSteps,N))
            xi = np.zeros((nTimeSteps-1,N,N))
            c_t = np.zeros(nTimeSteps)
            
            # Forward procedure (alpha)
            alpha[0,:] = prior*B[z[0],:]
            c_t[0] = 1./np.sum(alpha[0,:])
            alpha[0,:] = alpha[0,:]*c_t[0]
            
            for t in xrange(0,nTimeSteps-1):
                alpha[t+1,:] = B[z[t+1],:]*np.sum(T*alpha[t,:], axis=1)
                c_t[t+1] = 1./max(0.0000001,np.sum(alpha[t+1,:]))
                alpha[t+1,:] = alpha[t+1,:]*c_t[t+1]
            
            # Backward procedure (beta)
            beta[nTimeSteps-1,:] = 1
            beta[nTimeSteps-1,:] = beta[nTimeSteps-1,:]*c_t[nTimeSteps-1]
            
            for t in xrange(nTimeSteps-2,-1,-1):
                for i in xrange(0,N):
                    beta[t,i] = np.sum(B[z[t+1],:]*T[:,i]*beta[t+1,:])
                beta[t,:] = beta[t,:] * c_t[t]
                
            # forward-backward & pair of states (gamma and xi)
            for t in xrange(0,nTimeSteps-1):
                gamma[t,:] = alpha[t,:]*beta[t,:] / np.sum(alpha[t,:]*beta[t,:])
                
                for i in xrange(0,N):
                    xi[t,i,:] = alpha[t,:]*T[i,:]*B[z[t+1],i]*beta[t+1,i] # need to be divided by normalization constant
                xi[t,:,:] = xi[t,:,:] / np.sum(xi[t,:,:])
                
            gamma[nTimeSteps-1,:] = alpha[nTimeSteps-1,:]*beta[nTimeSteps-1,:] \
                                            / np.sum(alpha[nTimeSteps-1,:]*beta[nTimeSteps-1,:])
#            gamma = alpha*beta / np.sum(alpha*beta, axis=1)
            
            alphaList.append(alpha)
            betaList.append(beta)
            gammaList.append(gamma)
            xiList.append(xi)
        
        # Update model parameters (pi, T, B)
        
        gamma0combined = [ gammaList[k][0,:] for k in xrange(0,K) ]
        prior = sum(gamma0combined)/len(gamma0combined)
        
        # xi for 0:T time points across datasets (T_combined-5 x N x N)
        xiCombined = np.concatenate((xiList[0],xiList[1],xiList[2],xiList[3],xiList[4]),axis=0)
        
        # gamma for 0:T-1 time points across datasets (T_combined-5 x N)
        gammaCombined1 = np.concatenate((gammaList[0][:-1,:],gammaList[1][:-1,:],gammaList[2][:-1,:],\
                                         gammaList[3][:-1,:],gammaList[4][:-1,:]),axis=0)
        
        # gamma for 0:T time points across datasets (T_combined x N)
        gammaCombined2 = np.concatenate((gammaList[0],gammaList[1],gammaList[2],\
                                         gammaList[3],gammaList[4]),axis=0)
        
        T = np.sum(xiCombined,axis=0) / np.sum(gammaCombined1,axis=0)
        
        for uniqueZ in xrange(0,M):
            validInd = (movementSpecific_z_all == uniqueZ)
            B[uniqueZ, :] = np.sum(gammaCombined2[validInd,:],axis=0) / np.sum(gammaCombined2,axis=0)
        
        errorPrior = np.sum(np.abs(prior - prev_prior))
        errorT= np.sum(np.abs(T - prev_T))
        errorB = np.sum(np.abs(B - prev_B))
        
        error = np.max(np.array([errorPrior,errorT,errorB]))
        print('error = ' + str(error))
        
    return prior, T, B

def ViterbiDecoding(testData, total_kmeansObject, prior_list, T_list, B_list, N):
    nTimeSteps = testData[0].shape[0]

    deltaList = []
    prob_store = np.zeros(6)
    
    for movementIdx in xrange(0,6):
        
        kmeansObject = total_kmeansObject[movementIdx]
        z = kmeansObject.predict(testData[0][:,1:7])
    
        prior = prior_list[movementIdx]        
        B = B_list[movementIdx]
        T = T_list[movementIdx]
        
        delta = np.zeros((nTimeSteps, N))
        delta[0,:] = log(B[z[0],:]) + log(prior)
        
        for t in xrange(1,nTimeSteps):
            # max among row elements
            delta[t,:] = np.max( log(B[z[t],:]) + log(T) + delta[t-1,:], axis=1)
    
        deltaList.append(delta)
        prob_store[movementIdx] = np.max(delta[nTimeSteps-1], axis=0)
        
    return prob_store

def cutOffStationary(dataList):
    for i in xrange(0,len(dataList)):
        data = dataList[i]
        
        # first point where any accelerometer value is different by more than 2 from baseline
        ind1 = np.argmax(np.max(np.abs(np.mean(data[0:30,1:4],axis=0)-data[:,1:4]), axis=1) > 2)
        
        data = data[ind1:,:]
        
        # first point where all accelerometer values are different less than 2 away from baseline
#        ind2 = np.argmax( np.all( (np.abs(np.mean(data[-30:,1:4],axis=0)-data[:,1:4]) ) < 0.5, axis=1 ))
        dataFlip = np.flipud(data)
        
        ind2 = np.argmax(np.max(np.abs(np.mean(dataFlip[0:30,1:4],axis=0)-dataFlip[:,1:4]), axis=1) > 2)
        
        dataFlip = dataFlip[ind2:,:]
        
        data = np.flipud(dataFlip)
        
        dataList[i] = data
    
    return dataList

def normalizeData(dataList):
    for i in xrange(0,len(dataList)):
        data = dataList[i]

        # max and min within one axis of acceleration        
        accel_maxInterval = np.max( np.max(data[:,1:4]) - np.min(data[:,1:4]) )

        # max and min within one axis of gyro
        gyro_maxInterval = np.max( np.max(data[:,4:7]) - np.min(data[:,4:7]) )
        
        data[:,1:4] = data[:,1:4] / accel_maxInterval
        data[:,4:7] = data[:,4:7] / gyro_maxInterval
        
        dataList[i] = np.copy(data)
                   
    return dataList
    