#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time

# In[77]:


# Create class to handle "cities"
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
# Euclidean metric distance function
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# In[62]:


# Fitness function

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# ## Create our initial population

# In[4]:


# Route Generator

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# In[63]:


# Create initial population (route list)

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# ## Create the genetic algorithm

# In[64]:


# Ranking the routes

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


# In[65]:


#  Create a selection function that will be used to make the list of parent routes
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# In[66]:


# Create mating pool

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# In[67]:


# Create a crossover function for two parents to create one child

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# In[68]:


# Create function to run crossover over full mating pool

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


# In[69]:


# Create function to mutate a single route

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# In[70]:


# Create function to run mutation over entire population 

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# In[71]:


# Put all steps together to create the next generation

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# In[72]:


# Final step: create the genetic algorithm


    
def geneticAlgorithm(population,max_time, random_seed, input_GA, popSize, eliteSize, mutationRate, generations):
    random.seed(random_seed)
    start = time.time()
    trace = []
    pop = initialPopulation(popSize, population)
    fin_dist = []
    for i in range(0, generations):
        current = time.time()
        if (current-start<max_time):
            pop = nextGeneration(pop, eliteSize, mutationRate)
            fin_dist.append(int(1 / rankRoutes(pop)[0][1]))
            if(len(fin_dist)>1):
                if(fin_dist[-1]<fin_dist[-2]):
                    trace.append([time.time()-start,fin_dist[-1]])
        else:
            break
    
    final_dist = int(1 / rankRoutes(pop)[0][1])
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    b_r = bestRoute
    best_route_co_GA = []
    for i in range(len(b_r)):
        a = str(b_r[i]).split('(')
        a = a[1].split(',')
        a = [float(a[0].split('.')[0]), float(a[1].split(')')[0].split('.')[0])]
        best_route_co_GA.append(a)
    best_route_GA = []
    for q in best_route_co_GA:
        for p in input_GA:
            if(q[0]==p[1]):
                if(q[1]==p[2]):
                    print('hi')
                    best_route_GA.append(int(p[0]-1))

    return final_dist, best_route_GA, trace

def runGA(instance, max_time, random_seed):
    n_GA=int(open(instance).readlines()[2][len('DIMENSION: '):])
	
    input_GA=np.loadtxt(instance,skiprows=5,max_rows=n_GA)
    input_xy_GA = []
    for i in input_GA:
    	input_xy_GA.append((i[1], i[2]))
    cityList = []
    for i in range(0,len(input_xy_GA)):
        cityList.append(City(x=input_xy_GA[i][0], y=input_xy_GA[i][1]))

	
    min_dist, route, trace = geneticAlgorithm(population=cityList, max_time = max_time, random_seed = random_seed, input_GA=input_xy_GA, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
    return min_dist, route, trace

# ## Running the genetic algorithm

# In[73]:


# Create list of cities

# instance='Atlanta'
# n=int(open('../data/'+instance+'.tsp').readlines()[2][len('DIMENSION: '):])
# input=np.loadtxt("../data/"+instance+".tsp",skiprows=5,max_rows=n)


# # In[74]:


# input_xy = []
# for i in input:
#     input_xy.append((i[1], i[2]))


# # In[59]:


# cityList = []

# for i in range(0,len(input_xy)):
#     cityList.append(City(x=input_xy[i][0], y=input_xy[i][1]))


# In[75]:


# Run the genetic algorithm

# geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500, max_time =100)

