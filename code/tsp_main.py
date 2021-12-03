import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix
from optparse import OptionParser
import GA
import argparse
import SA
import Approx
import BnB

parser = argparse.ArgumentParser(description='TSP Solver Team 22')
parser.add_argument('-inst', nargs="?")
parser.add_argument('-alg', nargs="?")
parser.add_argument('-time', type=int)
parser.add_argument('-seed', type=int)


args = parser.parse_args()
method = args.alg
instance = args.inst.split(".")[0]
max_time = args.time
random_seed = args.seed

if(method=='GA'):

	n_GA=int(open(../data/instance).readlines()[2][len('DIMENSION: '):])
	input_GA=np.loadtxt(instance,skiprows=5,max_rows=n)
	input_xy_GA = []
	for i in input_GA:
		input_xy_GA.append((i[1], i[2]))
	cityList = []
	for i in range(0,len(input_xy_GA)):
		cityList.append(City(x=input_xy_GA[i][0], y=input_xy_GA[i][1]))

	min_dist, route, trace = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500, max_time = max_time, random_seed = random_seed)


elif(method == 'BnB'):
	input=np.loadtxt(instance,skiprows=5,max_rows=n)
	G = nx.from_numpy_matrix(np.matrix.round(distance_matrix(input[:,1:],input[:,1:])).astype(np.int32))
	min_dist, route, trace, runtime = BnB(G,max_time)

elif(method == 'Approx'):
	input=np.loadtxt(instance,skiprows=5,max_rows=n)
	G = nx.from_numpy_matrix(np.matrix.round(distance_matrix(input[:,1:],input[:,1:])).astype(np.int32))
	min_dist, route, trace, runtime = Approx(G,max_time)

elif(method == 'SA'):
	# complete-code
	n_SA=int(open(instance).readlines()[2][len('DIMENSION: '):])
	input_SA=np.loadtxt(instance,skiprows=5,max_rows=n)
	input_xy_SA = []
	for i in input_SA:
		input_xy_SA.append((i[1], i[2]))
	cityList = []
	for i in range(0,len(input_xy_SA)):
		cityList.append(City(x=input_xy_SA[i][0], y=input_xy_SA[i][1]))

	min_dist, route, trace = simanneal(cityList, max_iter = 100000)

trace = [",".join(map(str,x)) for x in trace]
tour_data = []
tour_data.append(min_dist)
tour_data.append(",".join(map(str,route)))  

outname = "./output/" +  file_name + "_" + method + "_" + str(max_time) + "_" + str(random_seed)

f_out = open(outname + ".tour", "w")
write_output(f_out,tour_data)
f_out.close()
f_out_tr = open(outname + ".trace", "w")
write_output(f_out_tr, trace_data)
f_out_tr.close()


	
