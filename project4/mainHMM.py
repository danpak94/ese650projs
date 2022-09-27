# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:22:41 2017

@author: Daniel
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
import proj4functions as p4f
import time
from sklearn.cluster import KMeans
import pdb
import os

#%%

data1 = np.loadtxt('../Proj4_2017_train/beat3_01.txt')
data2 = np.loadtxt('../Proj4_2017_train/beat3_02.txt')
data3 = np.loadtxt('../Proj4_2017_train/beat3_03.txt')
data4 = np.loadtxt('../Proj4_2017_train/beat3_06.txt')
data5 = np.loadtxt('../Proj4_2017_train/beat3_08.txt')

beat3_dataList = [data1,data2,data3,data4,data5]
beat3_dataList = p4f.cutOffStationary(beat3_dataList)
beat3_dataList = p4f.normalizeData(beat3_dataList)
beat3_dataCombined = np.concatenate((beat3_dataList[0],beat3_dataList[1],beat3_dataList[2],beat3_dataList[3],beat3_dataList[4]))

data1 = np.loadtxt('../Proj4_2017_train/beat4_01.txt')
data2 = np.loadtxt('../Proj4_2017_train/beat4_03.txt')
data3 = np.loadtxt('../Proj4_2017_train/beat4_05.txt')
data4 = np.loadtxt('../Proj4_2017_train/beat4_08.txt')
data5 = np.loadtxt('../Proj4_2017_train/beat4_09.txt')

beat4_dataList = [data1,data2,data3,data4,data5]
beat4_dataList = p4f.cutOffStationary(beat4_dataList)
beat4_dataList = p4f.normalizeData(beat4_dataList)
beat4_dataCombined = np.concatenate((beat4_dataList[0],beat4_dataList[1],beat4_dataList[2],beat4_dataList[3],beat4_dataList[4]))

data1 = np.loadtxt('../Proj4_2017_train/circle12.txt')
data2 = np.loadtxt('../Proj4_2017_train/circle13.txt')
data3 = np.loadtxt('../Proj4_2017_train/circle14.txt')
data4 = np.loadtxt('../Proj4_2017_train/circle17.txt')
data5 = np.loadtxt('../Proj4_2017_train/circle18.txt')

circle_dataList = [data1,data2,data3,data4,data5]
circle_dataList = p4f.cutOffStationary(circle_dataList)
circle_dataList = p4f.normalizeData(circle_dataList)
circle_dataCombined = np.concatenate((circle_dataList[0],circle_dataList[1],circle_dataList[2],circle_dataList[3],circle_dataList[4]))

data1 = np.loadtxt('../Proj4_2017_train/eight01.txt')
data2 = np.loadtxt('../Proj4_2017_train/eight02.txt')
data3 = np.loadtxt('../Proj4_2017_train/eight04.txt')
data4 = np.loadtxt('../Proj4_2017_train/eight07.txt')
data5 = np.loadtxt('../Proj4_2017_train/eight08.txt')

eight_dataList = [data1,data2,data3,data4,data5]
eight_dataList = p4f.cutOffStationary(eight_dataList)
eight_dataList = p4f.normalizeData(eight_dataList)
eight_dataCombined = np.concatenate((eight_dataList[0],eight_dataList[1],eight_dataList[2],eight_dataList[3],eight_dataList[4]))

data1 = np.loadtxt('../Proj4_2017_train/inf11.txt')
data2 = np.loadtxt('../Proj4_2017_train/inf13.txt')
data3 = np.loadtxt('../Proj4_2017_train/inf16.txt')
data4 = np.loadtxt('../Proj4_2017_train/inf18.txt')
data5 = np.loadtxt('../Proj4_2017_train/inf112.txt')

inf_dataList = [data1,data2,data3,data4,data5]
inf_dataList = p4f.cutOffStationary(inf_dataList)
inf_dataList = p4f.normalizeData(inf_dataList)
inf_dataCombined = np.concatenate((inf_dataList[0],inf_dataList[1],inf_dataList[2],inf_dataList[3],inf_dataList[4]))

data1 = np.loadtxt('../Proj4_2017_train/wave01.txt')
data2 = np.loadtxt('../Proj4_2017_train/wave02.txt')
data3 = np.loadtxt('../Proj4_2017_train/wave03.txt')
data4 = np.loadtxt('../Proj4_2017_train/wave05.txt')
data5 = np.loadtxt('../Proj4_2017_train/wave07.txt')

wave_dataList = [data1,data2,data3,data4,data5]
wave_dataList = p4f.cutOffStationary(wave_dataList)
wave_dataList = p4f.normalizeData(wave_dataList)
wave_dataCombined=np.concatenate((wave_dataList[0],wave_dataList[1],wave_dataList[2],wave_dataList[3],wave_dataList[4]))



total_dataCombined = np.concatenate((beat3_dataList[0],beat3_dataList[1],beat3_dataList[2],beat3_dataList[3],beat3_dataList[4],\
                                     beat4_dataList[0],beat4_dataList[1],beat4_dataList[2],beat4_dataList[3],beat4_dataList[4],\
                                     circle_dataList[0],circle_dataList[1],circle_dataList[2],circle_dataList[3],circle_dataList[4],\
                                     eight_dataList[0],eight_dataList[1],eight_dataList[2],eight_dataList[3],eight_dataList[4],\
                                     inf_dataList[0],inf_dataList[1],inf_dataList[2],inf_dataList[3],inf_dataList[4],\
                                     wave_dataList[0],wave_dataList[1],wave_dataList[2],wave_dataList[3],wave_dataList[4]),axis=0)


#%%

N = 10  # number of possible states (discrete)
M = 30  # number of possible observations (discrete)

K=5 # number of datasets per arm movement

# discrete measurement space

# list with each item as data for each gesture, combined through k datasets
total_dataCombinedList = [beat3_dataCombined, beat4_dataCombined, circle_dataCombined, \
                  eight_dataCombined, inf_dataCombined, wave_dataCombined]

# list with each item as data for each gesture for each k datasets
total_dataList = [beat3_dataList, beat4_dataList, circle_dataList, \
                  eight_dataList, inf_dataList, wave_dataList]

total_z_all_list = []
total_kmeansObject = []
total_zList = []

# movementIdx --> 0: beat3, 1: beat4, 2: circle, 3: eight, 4: inf, 5: wave
for movementIdx in xrange(0,6):
    kmeansObject = KMeans(n_clusters=M).fit(total_dataCombinedList[movementIdx][:,1:7])
    z_all = kmeansObject.labels_
    
    total_z_all_list.append(z_all)
    total_kmeansObject.append(kmeansObject)

    zList = p4f.getMovementSpecificVars(movementIdx, K, total_z_all_list[movementIdx], total_dataList[movementIdx])
    total_zList.append(zList)
    
#plt.plot(z_all,'.')
#plt.hist(z_all, bins=30)

#%% Main

start_time = time.time()

prior_list = []
T_list = []
B_list = []

for movementIdx in xrange(0,6):
    
    print('movementIdx = ' + str(movementIdx))
    
    #### Initial parameters (T,B,prior)
    prior, T, B = p4f.getInitializedVars(N, M)
    
    #### Baum Welch to get updated prior, T, and B
    prior, T, B = p4f.BaumWelch(prior, T, B, total_dataList[movementIdx], total_zList[movementIdx], \
                                total_z_all_list[movementIdx], N, M, K)
    
    prior_list.append(prior)
    T_list.append(T)
    B_list.append(B)

print("--- %s seconds ---" % (time.time() - start_time))

#%% Viterbi Decoding (testing model parameters)

testData = np.loadtxt('../Proj4_2017_train/circle12.txt')
#testData = np.loadtxt('../Proj4_2017_test/single/test1.txt')
testData = p4f.cutOffStationary([testData])
testData = p4f.normalizeData(testData)

prob_store = p4f.ViterbiDecoding(testData, total_kmeansObject, prior_list, T_list, B_list, N)

# movementIdx --> 0: beat3, 1: beat4, 2: circle, 3: eight, 4: inf, 5: wave
printMovement = {0: 'beat3', 1: 'beat4', 2: 'circle', 3: 'eight', 4: 'inf', 5: 'wave'}
print(printMovement[np.argmax(prob_store)])
print(prob_store)

#%% Vitervi Decoding (all data files in one folder)

#testData_dir = '../Proj4_2017_test/multiple/'
testData_dir = '../Proj4_2017_train/'

savePredictions = {}

printMovement = {0: 'beat3', 1: 'beat4', 2: 'circle', 3: 'eight', 4: 'inf', 5: 'wave'}

for filename in os.listdir(testData_dir):
    testData = np.loadtxt(testData_dir+filename)
    testData = p4f.cutOffStationary([testData])
    testData = p4f.normalizeData(testData)
    prob_store = p4f.ViterbiDecoding(testData, total_kmeansObject, prior_list, T_list, B_list, N)
    
    savePredictions[filename] = printMovement[np.argmax(prob_store)]
    
print(savePredictions)


