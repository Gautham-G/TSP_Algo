import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix
from optparse import OptionParser
from GA import *
import argparse
from SA import *
import Approx
import BnB
import sys

parser = argparse.ArgumentParser(description='TSP Solver Team 22')
parser.add_argument('-inst', nargs="?", type=str)
parser.add_argument('-alg', nargs="?")
parser.add_argument('-time', type=int)
parser.add_argument('-seed', type=int)


args = parser.parse_args()
method = args.alg
instance = args.inst
max_time = args.time
random_seed = args.seed

def write_output(file_ptr,datalines):
    for i in datalines:
        file_ptr.write(str(i) + "\n")


if(method=='GA'):	
	min_dist, route, trace = runGA(instance, max_time, random_seed)
	print(type(route))

elif(method == 'BnB'):
	n=int(open(instance).readlines()[2][len('DIMENSION: '):])
	input=np.loadtxt(instance,skiprows=5,max_rows=n)
	G = nx.from_numpy_matrix(np.matrix.round(distance_matrix(input[:,1:],input[:,1:])).astype(np.int32))
	min_dist, route, trace, runtime = BnB(G,max_time)

elif(method == 'Approx'):
	n=int(open(instance).readlines()[2][len('DIMENSION: '):])
	input=np.loadtxt(instance,skiprows=5,max_rows=n)
	G = nx.from_numpy_matrix(np.matrix.round(distance_matrix(input[:,1:],input[:,1:])).astype(np.int32))
	min_dist, route, trace, runtime = Approx(G,max_time)

elif(method == 'SA'):
	# complete-code
	n_SA=int(open(instance).readlines()[2][len('DIMENSION: '):])
	input_SA=np.loadtxt(instance,skiprows=5,max_rows=n_SA)
	input_xy_SA = []
	for i in input_SA:
		input_xy_SA.append((i[1], i[2]))
		
	sa = simanneal(input_xy_SA, max_time)
	min_dist, route, trace = sa.executeanneal()

trace = [",".join(map(str,x)) for x in trace]
tour_data = []
tour_data.append(min_dist)
tour_data.append(",".join(map(str,route)))  

file_name = instance
outname = "../output/" +  file_name + "_" + method + "_" + str(max_time) + "_" + str(random_seed)

f_out = open(outname + ".tour", "w")
write_output(f_out,tour_data)
f_out.close()
f_out_tr = open(outname + ".trace", "w")
write_output(f_out_tr, trace)
f_out_tr.close()


	
