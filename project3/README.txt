
Daniel Pak

Uploaded files:
1. mainSLAM.py
2. proj3functions.py
3. load_data.py
4. p3_util.py
5. implementRGBD.py 

Explanation on how to run:

1. mainSLAM.py

Input file directory for lidar and joint data, and run the script. Currently, it calculates every 10 time points. You can change this by adjusting the "increment" variable. The printed numbers are time indices.

You can also change the grid size by changing "xs_size" and "ys_size" to 9 or 1. If you want to change to different number of grids (must be odd number), xs and ys have to be changed manually. The spacing of grid points can also be changed with xs and ys. The only requirement is that they need to be integer arrays and the middle point is 0.

You can change N_eff varialbe to change the frequency of resampling.

2. proj3functions.py

File with all the functinos that do the calculations for mainSLAM.

You can change:
	Noise (x,y,theta)
	mapCorrelation function to measure 9x9 grid or single point (need to uncomment block of code and comment existing code)

3. load_data.py

This file is neded to load data properly.

4. p3_util.py

This file is included just in case (to avoid errors).

5. implementRGBD.py

Run mainSLAM first to get SLAM map. This file works under the assumption that the MAP variable is existing in the console. Input file directory and run this script to create a textured map.
