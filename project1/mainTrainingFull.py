# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:47:29 2017

@author: Daniel
"""

import cv2
import os
import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

fileSaveName='zzz'

dataFileName='trainingPixels_stored'
trainingPixels_stored=np.load(dataFileName+'.npy')

folder="C:\Users\Daniel\Google Drive\ESE650\project 1\Training_set"
#folder="D:\Daniel\Google Drive\ESE650\project 1\Training_set"

colorSpace=cv2.COLOR_RGB2HSV
trainingPixels = np.concatenate(trainingPixels_stored[:])
trainingPixels3D = np.expand_dims(trainingPixels,axis=1)
trainingPixels3D = cv2.cvtColor(np.uint8(trainingPixels3D),colorSpace)

nClusters=3

trainingPixels2D = np.squeeze(trainingPixels3D,axis=1)
kmeans = KMeans(n_clusters=nClusters).fit(trainingPixels2D)

pixelsInCluster=[]
covCluster=[]
meanCluster=[]
pdfVals=[]
membership=[]

# Initializing membership probabilities
for x in range(0,nClusters):
    pixelsInCluster.append(trainingPixels2D[kmeans.labels_==x])
    covCluster.append(np.cov(pixelsInCluster[x].T))
    meanCluster.append(kmeans.cluster_centers_[x,:])
    
    I=np.copy(trainingPixels2D)
    mu=meanCluster[x]
    mu=np.asarray(mu)
    L=np.linalg.cholesky(np.linalg.inv(covCluster[x]))
    intermediate=np.dot(I-mu,L)
    g=np.sum(np.square(intermediate),axis=1)
    
    pdfVals.append(np.exp(-1/2*g))

#==============================================================================
# Checking pixel distributoin and kMeans centroids
selectPixels=np.random.permutation(trainingPixels2D)
idx=np.linspace(0,len(trainingPixels2D)-1,1000)
idx=idx.astype(int)
selectPixels=selectPixels[idx]
fig=plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
ax.scatter(selectPixels[:,0],selectPixels[:,1],selectPixels[:,2])
ax.scatter(zip(*meanCluster)[0],zip(*meanCluster)[1],zip(*meanCluster)[2],c='red',s=80)
plt.show()
#==============================================================================

a_k = 1/float(nClusters)*np.ones((nClusters,1))
normalizationMembership = np.sum(a_k*pdfVals,axis=0)
for x in range(0,nClusters):
    membership.append(a_k[x]*pdfVals[x]/normalizationMembership)
membership=np.asarray(membership)

#==============================================================================
# Making sure membership probabilities and coefficients sum to 1
#print(a_k)
#print(normalizationMembership)
#print(np.sum(membership,axis=0))
#==============================================================================

#==============================================================================
# Testing EM algorithm w/o for loop for clusters
#x=0   #cluster number
#mu_k=np.empty((nClusters,3))
#mu_k[x,:]=np.dot(membership[x,:],trainingPixels2D)/np.sum(membership[x,:])
#sigma_k=np.empty((nClusters,3,3))
#diagMatrix=[]
#for y in range(0,len(trainingPixels2D)):
#    diagMatrix.append(np.diag(trainingPixels2D[y,:]-mu_k[x,:]))
#
#diagMatrix=np.multiply(diagMatrix,diagMatrix)
#sigma_k=np.tensordot(membership[x,:],diagMatrix,axes=1)/np.sum(membership[x,:])
#print(np.shape(diagMatrix))
#print(sigma_k)
#print(covCluster[x])
#==============================================================================

diffMean=100*np.ones((nClusters,1))
diffCov=100*np.ones((nClusters,3))
mu_k=np.empty((nClusters,3))
sigma_k=np.empty((nClusters,3,3))

dummyMean=meanCluster
dummyCov=covCluster
print('meanCluster')
print(meanCluster)
print('covCluster')
print(covCluster)
counter=0
while np.amax(np.absolute(diffMean)) > 1e-2 or np.amax(np.absolute(diffCov)) > 1e-1:
    for x in range(0,nClusters):
        mu_k[x,:]=np.dot(membership[x,:],trainingPixels2D)/np.sum(membership[x,:])
        
        diagMatrix=[]
        for y in range(0,len(trainingPixels2D)):    
            diagMatrix.append(np.diag(trainingPixels2D[y,:]-mu_k[x,:]))
        diagMatrix=np.multiply(diagMatrix,diagMatrix)
        
        sigma_k[x,:,:]=np.tensordot(membership[x,:],diagMatrix,axes=1)/np.sum(membership[x,:])
        a_k[x]=1/float(len(trainingPixels2D))*np.sum(membership[x,:])
        
        I=np.copy(trainingPixels2D)
        L=np.linalg.cholesky(np.linalg.inv(sigma_k[x,:,:]))
        intermediate=np.dot(I-mu_k[x,:],L)
        g=np.sum(np.square(intermediate),axis=1)
        
        pdfVals[x]=np.exp(-1/2*g)
        
    normalizationMembership = np.sum(a_k*pdfVals,axis=0)
    for x in range(0,nClusters):
        membership[x,:]=a_k[x]*pdfVals[x]/normalizationMembership
    
    diffMean=np.sum(mu_k-dummyMean,axis=1)
    diffCov=np.sum(sigma_k-dummyCov,axis=0)
    dummyMean=np.copy(mu_k)
    dummyCov=np.copy(sigma_k)
    counter=counter+1
    print(counter)
    print(diffMean)
    print(diffCov)
#    print(dummyMean)
#    print(dummyCov)
#    print(mu_k)
#    print(sigma_k)

# Plotting some of the training points and final mu_k
selectPixels=np.random.permutation(trainingPixels2D)
idx=np.linspace(0,len(trainingPixels2D)-1,1000)
idx=idx.astype(int)
selectPixels=selectPixels[idx]
fig=plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
ax.scatter(selectPixels[:,0],selectPixels[:,1],selectPixels[:,2])
ax.scatter(mu_k[:,0],mu_k[:,1],mu_k[:,2],c='red',s=200)
plt.show()

np.save(fileSaveName+'-mu_k',mu_k)
np.save(fileSaveName+'-sigma_k',sigma_k)
np.save(fileSaveName+'-a_k',a_k)
#np.save('trainInd',trainInd)
#np.save('testInd',testInd)


print('done')

