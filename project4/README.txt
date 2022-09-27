Daniel Pak
03-24-2017

Uploaded files:
1. mainHMM.py
2. submainHMM.py
3. proj4functions.py
4. .pickle files

Explanation on how to run:

1. mainHMM.py

This is the main script for training and testing the HMM. User must change directories to load the right files for training and testing.

Running the script will result in multiple things printing in the console:
a.) movement index (each number corresponding to a gesture)
b.) error value in each iteratino of Baum-Welch algorithm
c.) time it took to complete the Baum-Welch algorithm
d.) single classification for one specified testset (useful for specific debugging)
e.) multiple classification for all testsets in a specified folder (useful for overall accuracy)

2. submainHMM.py

This file contains multiple sections that provide various visualizaion / debugging utilities.

3. proj4functions.py

This file contains all functions required to run the mainHMM.py script.

4. .pickle files

These are the model parameter files that I used to get the training and testing results.
