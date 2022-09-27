# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:19:17 2017

@author: Daniel
"""



#%% Visualize data

#data = beat3_dataList[0]
#data = wave_dataList[3]
#data = testData[0]
data = np.loadtxt('../Proj4_2017_train/beat3_31.txt')

plt.clf()

plt.subplot(211)
plt.plot(data[:,1], label='Ax')
plt.plot(data[:,2], label='Ay')
plt.plot(data[:,3], label='Az')
plt.legend()

plt.subplot(212)
plt.plot(data[:,4], label='Wx')
plt.plot(data[:,5], label='Wy')
plt.plot(data[:,6], label='Wz')
plt.legend()


#%% Visualize transition / observation matrices, and number of each observation z

fig, ax = plt.subplots()
im = plt.imshow(B_list[0])
fig.colorbar(im)


num_uniqueZ = np.zeros(M)
for uniqueZ in xrange(0,M):
    num_uniqueZ[uniqueZ] = np.sum(total_z_all_list[0] == uniqueZ)

plt.figure()
plt.plot(num_uniqueZ)


#%% Needed if I want to run function in p4f separately

movementSpecific_dataList = total_dataList[movementIdx]
movementSpecific_zList = total_zList[movementIdx]
movementSpecific_z_all = total_z_all_list[movementIdx]


#%% Save model params

import pickle

with open('modelParams - yes_dataNormalization.pickle','w') as f:
    pickle.dump([prior_list,T_list,B_list, total_kmeansObject],f)

#%% Load model params

import pickle

with open('modelParams - yes_dataNormalization.pickle') as f:
    prior_list,T_list,B_list, total_kmeansObject = pickle.load(f)

