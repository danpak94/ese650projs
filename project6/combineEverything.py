# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:43:12 2017

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from scipy.io import loadmat
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import proj6functions as p6f

#%% 

#start = time.time()
#end = time.time()
#print(end - start)

fileName = '21'
RGBdata = skimage.io.imread('../terrain/rgb/'+fileName+'.png')
DEPTHdata = skimage.io.imread('../terrain/depth/'+fileName+'.png')

params = loadmat('../kv2_param.mat')

[P_rgb, u_rgb, v_rgb] = p6f.depth2rgb(DEPTHdata, params)
[colorList, handPickedIdxArray] = p6f.getImgColorList(RGBdata, u_rgb, v_rgb)

##%% neighborhood normal (takes a few seconds)

radius=50 # mm
[normalsList, normalPointsList, normalIdxArray] = p6f.getNeighborhoodNormal(radius, RGBdata, P_rgb, u_rgb, v_rgb)

##%% Clustering in normal space (takes a few seconds)

normalVectors = np.asarray(normalsList).squeeze() # n_samples x n_features
#bandwidth = estimate_bandwidth(normalVectors, n_samples=2000)
ms = MeanShift(bandwidth=0.11, bin_seeding=True)
ms.fit(normalVectors)

print(len(np.unique(ms.labels_)))
print(ms.cluster_centers_.shape)

normalColorList = p6f.getClusterColorList(ms, normalsList)

#%% Clustering in distance space for each set of normal vector cluster

combinedPointsList = []
combinedNormalsList = []
combinedColorList = []

for loopIndex in xrange(ms.cluster_centers_.shape[0]):

    normalClusterIdx = loopIndex
    
    idx = (ms.labels_== normalClusterIdx)
    subsetNormalPoints = P_rgb[:, normalIdxArray]
    subsetNormalPoints = subsetNormalPoints[:, idx].T
    msDist = MeanShift(bandwidth=200, bin_seeding=False)
    msDist.fit(subsetNormalPoints)
    
    print(len(np.unique(msDist.labels_)))
    
    subsetNormalsList = []
    for k in xrange(len(normalsList)):
        if idx[k]==True:
            subsetNormalsList.append(normalsList[k])
            
    subsetNormalPointsList = []
    for k in xrange(subsetNormalPoints.shape[0]):
        subsetNormalPointsList.append(subsetNormalPoints[k,:].reshape(-1,1))
    
    distColorList = p6f.getClusterColorList(msDist, subsetNormalsList)
    
    ##%%
    
    thresh_point2plane = 15 # mm
    thresh_dist_division_factor = 6
    
    ind_grouped = np.zeros(len(msDist.cluster_centers_), dtype=int)
    
    labels = np.copy(msDist.labels_)
    
    # loop starts here (iterating until no more clusters can be merged)
    # idx changes at the end of loop
    
    changed=1
    previous_labels = 1e6*np.ones(np.size(labels))
    
    for iterative in xrange(100):
        
    #    print('updating clusters')
        
        centroidArray = np.zeros((len(np.unique(labels)), 3))
        for k in xrange(len(np.unique(labels))):
            centroidArray[k,:] = np.mean(subsetNormalPoints[labels==np.unique(labels)[k],:], axis=0)
        
        ind_grouped = np.zeros(centroidArray.shape[0], dtype=int)
        not_processed_status = np.ones(ind_grouped.shape, dtype=bool)
        for k in xrange(len(np.unique(labels))):
    #    for k in xrange(10):
            singleCentroid = centroidArray[k,:]
            singleNormal = np.mean(normalVectors[ms.labels_== normalClusterIdx,:][labels==np.unique(labels)[k],:], axis=0).reshape(-1,1)
            pi_minus_c =  centroidArray - singleCentroid
            
            if np.sum(labels==np.unique(labels)[k])!=0:
                U,s,V = np.linalg.svd(subsetNormalPoints[labels==np.unique(labels)[k],:].T, full_matrices=True)
    #            print(s[0])
                thresh_dist = np.copy(s[0]/thresh_dist_division_factor)
            
            ind_point2plane = ( abs( np.dot( pi_minus_c, singleNormal )) < thresh_point2plane ).squeeze().astype(bool)
            
            ind_dist = ( np.sum( abs(centroidArray - singleCentroid), axis=1 ) < thresh_dist ).astype(bool)
            
            ind_grouped[(ind_point2plane & ind_dist) & not_processed_status] = np.unique(labels)[k]
            
            not_processed_status[ind_point2plane & ind_dist] = False
        
        # converting ind_grouped to a shape compatible with mean-shift labels
        newLabels = np.zeros(labels.shape)
        for k in xrange(len(np.unique(labels))):
            newLabels[labels==np.unique(labels)[k]] = ind_grouped[k]
    
        if np.array_equal(previous_labels, labels):
            changed=0
            
        previous_labels = np.copy(labels)
        labels = np.copy(newLabels)
        
    #idx = np.zeros((msDist.labels_).shape, dtype=int)
    #idx[msDist.labels_==4] = 4
    
    #plt.plot(idx, 'o')
    print(len(np.unique(labels)))
    
    groupedColorList = p6f.getGroupedColorList(msDist, subsetNormalsList, labels)
    
    combinedPointsList.append(subsetNormalPointsList)
    combinedNormalsList.append(subsetNormalsList)
    combinedColorList.append(groupedColorList)

#%%

import vtk

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=3e6):
        self.maxNumPoints = maxNumPoints
        
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        
        self.vtkLinePts = vtk.vtkPoints()
        self.vtkLine = vtk.vtkLine()
        self.vtkLineCell = vtk.vtkCellArray()
        
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
#        self.vtkPolyData.SetPoints(self.vtkLinePts)
        self.vtkPolyData.SetLines(self.vtkLineCell)
        
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")
        self.vtkPolyData.GetCellData().SetScalars(self.colors)
        self.vtkPolyData.GetPointData().SetActiveScalars('Colors')
        
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.vtkPolyData)
        self.mapper.SetColorModeToDefault()
        self.mapper.SetScalarRange(zMin, zMax)
        self.mapper.SetScalarVisibility(1)
        
#        arrowSource = vtk.vtkArrowSource()
#        mapper.SetInputConnection(arrowSource.GetOutputPort())
        
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(self.mapper)
        self.vtkActor.GetProperty().SetPointSize(4)
        self.axes = vtk.vtkAxesActor()
        self.vtkActor.GetProperty().SetLineWidth(5)

    def addPoint(self, point, colorTuple):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            self.colors.InsertNextTupleValue(colorTuple)
        else:
            print('too many points!')
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        
    def addLine(self, c, n, colorTuple):
        nPoints = self.vtkPoints.GetNumberOfPoints()
        
        self.vtkPoints.InsertNextPoint(c)
        self.vtkPoints.InsertNextPoint(c-n*50)
        self.vtkLine.GetPointIds().SetId(0,nPoints)
        self.vtkLine.GetPointIds().SetId(1,nPoints+1)
        self.vtkLineCell.InsertNextCell(self.vtkLine)
        self.colors.InsertNextTupleValue(colorTuple)
        
        
pointCloud = VtkPointCloud()
for k in xrange(P_rgb.shape[1]):
    pointCloud.addPoint(P_rgb[:,k], colorList[k])

#for k in xrange(len(normalsList)):
##    if ms.labels_[k]==1:
#    pointCloud.addLine(normalPointsList[k], normalsList[k], normalColorList[k])

#for k in xrange(len(distColorList)):
#    pointCloud.addLine(subsetNormalPointsList[k], subsetNormalsList[k], distColorList[k])

#for k in xrange(len(groupedColorList)):
#    pointCloud.addLine(subsetNormalPointsList[k], subsetNormalsList[k], groupedColorList[k])

for j in xrange(len(combinedColorList)):
    subsetNormalPointsList = combinedPointsList[j]
    subsetNormalsList = combinedNormalsList[j]
    groupedColorList = combinedColorList[j]
    
    for k in xrange(len(groupedColorList)):
        pointCloud.addLine(subsetNormalPointsList[k], subsetNormalsList[k], groupedColorList[k])

camera = vtk.vtkCamera()
camera.SetPosition(0, 0, 0);
camera.SetFocalPoint(0, 0, 1);
camera.Roll(180)

# Renderer
renderer = vtk.vtkRenderer()
renderer.SetActiveCamera(camera)
renderer.AddActor(pointCloud.vtkActor)
#renderer.AddActor(pointCloud.axes)

cubeAxesActor = vtk.vtkCubeAxesActor()
cubeAxesActor.SetBounds(pointCloud.vtkPolyData.GetBounds())
cubeAxesActor.SetCamera(renderer.GetActiveCamera())
cubeAxesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
cubeAxesActor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)
cubeAxesActor.DrawXGridlinesOn()
cubeAxesActor.DrawYGridlinesOn()
cubeAxesActor.DrawZGridlinesOn()

#renderer.AddActor(cubeAxesActor)
renderer.SetBackground(.2, .3, .4)
#renderer.SetBackground(1, 1, 1)
renderer.ResetCamera()
renderer.GetActiveCamera().Zoom(2)

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
renderWindow.Render()
renderWindowInteractor.Start()

del renderWindow, renderWindowInteractor
