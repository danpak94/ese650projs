# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:21:34 2017

@author: Daniel
"""

import numpy as np
from numpy import cos,sin
import pickle
import proj3functions as p3f
import load_data as ld
import matplotlib.pyplot as plt

#%%

with open('../Proj3_2017_Train_rgb/cameraParam/exParams.pkl') as f:
    RTparam=pickle.load(f)
    
with open('../Proj3_2017_Train_rgb/cameraParam/IRcam_Calib_result.pkl') as f:
    IRparam=pickle.load(f)

with open('../Proj3_2017_Train_rgb/cameraParam/RGBcamera_Calib_result.pkl') as f:
    RGBparam=pickle.load(f)

#%%

RGB = ld.get_rgb('../Proj3_2017_Train_rgb/RGB_0')
IR = ld.get_depth('../Proj3_2017_Train_rgb/DEPTH_0')

lidar_data=ld.get_lidar('../Proj3_2017_Train/data/train_lidar0')
joint_data=ld.get_joint('../Proj3_2017_Train/data/train_joint0')

#idxLidar,idxJoint=p3f.getTimeInds(lidar0,joint0)
#%%

idxLidar2,idxJoint2,idxRGB, idxIR = p3f.getTimeInds2(lidar_data,joint_data,RGB,IR)


#%%

heightCamera = 0.07+0.33+0.93

HeadAngle = joint_data['head_angles'][1,idxJoint2] # pitch
theta = np.copy(HeadAngle)

n_ground = np.zeros((theta.size,3),dtype=float)
for i in xrange(0,theta.size):
    n_ground[i,:] = np.dot(np.array([[cos(theta[i]),-sin(theta[i]),0],\
                                 [sin(theta[i]),cos(theta[i]),0], \
                                 [0,0,1]]), \
                      np.array([0,1,0]).T)

a0 = n_ground[:,0]
a1 = n_ground[:,1]
a2 = n_ground[:,2]
a3 = -heightCamera

#plane = a0*u+a1*v+a2+a3*d

#%%

RGBprincipalPt = RGBparam['cc']
RGBimage = [rgb['image'] for rgb in RGB]

IRprincipalPt = IRparam['cc']
IRdepth = [ir['depth'] for ir in IR]

#