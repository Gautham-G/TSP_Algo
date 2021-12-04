# CSE 6140 - TSP project

Project Team:
* Gautham Gururajan
* Vishal Hariharan
* Aranya Banerjee 

Analysis of Algorithms to solve the TSP


## OS

* Linux (Ubuntu 20.04) 

## Dependencies

* networkx==2.6.3
* numpy==1.19.5
* pandas==1.1.5
* scipy==1.4.1
* random
* queue
* optparse
* matplotlib
* sys
* os
* math
* operator


## Command to run 

* (1) python3 tsp_main.py -inst <filename> -alg [BnB | Approx | LS1 | LS2] -time <cutoff_in_seconds> -seed <random_seed>
For example,
python3 tsp_main.py -inst ../data/Atlanta.tsp -alg GA -time 30 -seed 1

