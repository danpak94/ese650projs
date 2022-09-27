Daniel Pak

Uploaded files:
1. main_vtk.py
2. combinedEverything.py
3. proj6functions

Dependencies:
1. vtk (visualization toolkit) - installed using anaconda
2. sklearn - installed using anaconda
3. skimage.io -installed using anaconda

Other standard ones like numpy, matplotlib, scipy, time, etc.

Explanation on how to run:

1. main_vtk.py

Specify number of file with "fileName" variable. Code assumes the data folder is in "terrain" folder, located in the same level directory as the script folder.

Parameters you can adjust:
	a. radius - radius defining the neighborhood of points
	b. bandwidth (as MeanShift argument) - bandwidth for mean shift algorithm. Lower = more clusters
	c. normalClusterIdx - only change if looking at individual normal clusters
	d. thresh_point2plane - used for post-merging after normal and distance clustering. Higher = less precision for plane height discrimination
	e. thresh_dist_division_factor - also used for post-merging. Higher = smaller radius for merge considerations

After adjusting the parameters (if desired), run the script to process the data and plot with vtk. Once the OpenGL window opens, you can navigate within the window with mouse left click, right click, shift + left click, ctrl + left click, mouse wheel, keyboard press "r", and more. To quit the window, keyboard press "q" or click the X.

2. combinedEverything.py

The purpose of this file is to combine all processes tested in main_vtk with a single run of the file. It was used for creating the full final results for the presentation.

3. proj6functions.py

This file includes all functions used for this project. meanshiftDP is my attempt at implementing the mean shift algorithm. I am having some issues with the final positioning of the cluster centers, so it is still being debugged as of the time of code submission.
