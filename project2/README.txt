
Daniel Pak 02-21-2017

Note: I assume the data files are in folders like cam/cam1, etc. and the .py files are one directory above those folders.

Uploaded files:
1. mainUKF-7sigmaPts.py
2. proj2functions.py
3. eulerangles.py
4. imageStitch.py
5. imageStitch-checkRotations.py
6. rotplot.py
7. finalCombined.py

Explanation on how to run:

1. mainUKF-7sigmaPts.py

User must input an integer for loading different files. The variable name is fileIdx (at the top of the file), and changing this number will change the files that the script imports to run the UKF code (e.g. fileIdx=1 will import cam1, vicon1, imuraw1 data). The directory can be changed in the "filename" variables directly underneath. "filename1" is cam data (pixels), "filename2" is imu data, and "filename3" is viconRot data.

After inputting the file index, run the script. It will load the data, calculate the values of sensors in real world units, and do all the UKF calculations. The iteration number will be printed in Python console (iteration for each time point). To test the code's response to noise levels, adjust the variables "processNoise", which corresponds to Q in Kraft's paper, and "measurementNoise", which corresponds to R in Kraft's paper.

The plot at the end shows the comparisons of euler angles between Vicon rotation matrices (ground truth), estimates purely from gyroscope (no sigma points or UKF applied), and UKF output. The plots should be labeled accordingly.

2. proj2functions.py

This file contains all functions used for project 2. There is nothing to run in this code - I simply import this file to use in the mainUKF file.

3. eulerangles.py

Similar to proj2functions.py, I simply import some "built-in" functions from this file to use in the mainUKF file.

One important thing to note is that this file requires the library nibabel to run properly.

4. imageStitch.py

User must input an integer for fileIdx similar to mainUKF.py.

When you run this script, it will show the image stitching process in real-time. The number of images used for stitching can be changed with the variable "imageIncrement". Use only integers for this variable.

Note that this file uses the Vicon rotation matrices to do the stitching.

5. imageStitch-checkRotations.py

This file is used for making sure that the rotation matrices match with the image orientation. When you run this script, you will see a rotplot on one side and the raw image for the corresponding time point on the other side.

6. rotplot.py

This file is needed to run the imageStitch-checkRotations.py file. No need to run this.

7. finalCombined.py

This file combines the UKF approximation and image stitching. Change the fileIdx, similar to above, to try different files. It is literally a combination of the files mainUKF-7sigmaPts.py and imageStitch.py, so it will behave similarly as the files do separately - it will print the time index while calculating the UKF approximations, display the results, and then do the stitching in real-time.
