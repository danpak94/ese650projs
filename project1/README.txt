Daniel Pak

Note: I used the easier/2nd training set.

Uploaded files:
1. gettingTrainingPixels.py
2. mainTrainingCV.py
3. maingTrainingFull.py
4. mainTesting.py
5. .npy files, which include model parameters for barrelRed and notBarrelRed
6. roipoly.py file which is used to select regions for labeling

Explanation on how to run:

1. gettingTrainingPixels.py

User must input string values for "fileNameToBeSaved" and "folder".
The code will pull image files from "folder" and eventually all selected pixel values will be saved in fileNameToBeSaved+".npy" as a numpy array.

After inputting the string values, run the script. It will go through every image file in the specified "folder" and you will have to select regions of interest depending on what type of dataset this will be (barrel-red, not-barrel-red, other colors, etc.). Method of selecting regions is roipoly. When the image pops up, left-click to specify all vertices and right click to enclose the region. The mask of selected region will pop up to verify to the user that the region has been selected successfully.

2. mainTrainingCV.py

First thing is to specify "fileSaveName", which is the name of the .npy file to which the model parameters will be saved, and dataFileName", which contains the desired pixel values acquired from running the previous file "gettingTrainingPixels.py". Then, specify the "folder" where images are stored.

Since this is a file used for training for cross-validation purposes (hence the name CV), there are several lines that help randomly split up the indices for 10-fold cross-validation. However, since training the models is very time consuming, I have saved the initial train and test indices and simply load them from a .npy file. New train and test indices can be acquuired by uncommenting the section "Generate new train and test indices for cross validation".

You can change the variable "colorSpace" to try different color spaces.
You can change the variable "nClusters" to change the number of Gaussians in the Gaussian Mixture Model. 

Otherwise, simply run the script and it should automatically start training to find the best mean, covariance, and coefficients for GMM using the EM algorithm. In the middle, it will plot a figure to show the initial positions of the means in the midst of some pixels (randomly chosen from training set), which you will have to close to continue with the script. During the optimization, it will display the number of cycles, summed change in mean, and summed change in variance. Once the summed changes hit a certain threshold value (which you can adjust at the argument of the while loop), the algorithm will stop and display the final mean along with some random pixels from the training set.

3. mainTrainingFull.py

Essentially identical to mainTrainingCV.py, except with a few different lines to include the entire training set data instead of choosing a percentage of them for cross-validation purposes.

4. mainTesting.py

The only thing to specify in this script is the folder directory of the testing images (variable "folder"). The model parameters of barrel_red and not_barrel_red should all be loaded if the .npy files are in the same folder as the mainTesting.py file itself. All libraries are pretty standard (new to Python so I'm not exactly sure how I'm supposed to talk about "dependencies").

Running the script will iterate through every item in the folder and output an image with 3 subplots: color-segmented image, post-processed image (with openCV2), and the original image with bounding box(es). The values for the corners of the bounding box and the estimated distance of the barrel will be printed on the console.
