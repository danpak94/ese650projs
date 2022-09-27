Daniel Pak

Uploaded files:
1. Same exact files as the ones downloaded in Proj5_2017_train. The only difference is that I edited some functions to make them work for different parts of the project. 

Changed files:
1. 1_MDP
	a. todo/todo.py
	b. util/frozen_lake.py
	c. run_FrozenLake.py
2. 2_PGO
	a. Nothing yet.. (will attemp this part tonight and submit w/ late day if it works)

Explanation on how to run:

1a. todo/todo.py

	Nothing needs to be changed here. This file has all the updated #TODO functions.

1b. util_frozen_lake.py

	You can test 4x4 or 8x8 frozen lake maps by changing the "map_name" input in the constructor of class FrozenLakeEnv.

1c. run_FrozenLake.py

	Some parameters need to be changed depending on the size of the map.
	For 4x4:
		n_iter = 100, stepsize = 100 (arguments of stat2ts in function test_PGO)
	For 8x8:
		n_iter = 1000, stepsize = 200 (arguments of stat2ts in function test_PGO)
	For both:
		horizon = 50 (argument of FL.animate_rollout in function test_PGO)

Last step is running the entire script (run_FrozenLake.py). It will print various statistics, and then plot them over the number of iterations at the end of model training. It will also display an episode until the agent reaches a hole or the goal, or when the number of steps exceeds the "horizon" value.
