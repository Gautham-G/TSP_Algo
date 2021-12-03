import numpy as np
import sys
import random as rd
import math
import pandas as pd
import time

class simanneal(object):
    def __init__(self, points, max_time):
        
        self.points = points
        self.n = len(points)
        self.T = (self.n)**0.5
        self.alpha = 0.5
        self.min_T = 10**(-8)
        self.max_time = max_time
        self.iter = 1
        self.nodelist = [node for node in range(self.n)]
        self.best_dist = float("Inf")
        self.optimal_solution = None
        self.best_list = []
        
    def randompick(self):
        
        selected_node = rd.choice(self.nodelist)
        sol = [selected_node]
        rem_nodes = set(self.nodelist)
        rem_nodes.remove(selected_node)
        while rem_nodes:
            
            nrst_ngbr = min(rem_nodes, key = lambda x: self.eucdist(selected_node, x))
            rem_nodes.remove(nrst_ngbr)
            sol.append(nrst_ngbr)
            
        dist = self.totdist(sol)
        if dist < self.best_dist:
            self.bist_dist = dist
            self.optimal_solution = sol
            
        self.best_list.append(dist)
        
        return sol, dist
    
    def eucdist(self, node1, node2):
        
        point1, point2 = self.points[node1], self.points[node2]
        return int(np.rint(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5))
    
    
    def totdist(self, sol):
        
        dist = 0
        for i in range(self.n):
            dist += self.eucdist(sol[i%self.n], sol[(i+1)%self.n])
        return dist
    
    def executeanneal(self):
        
        start = time.time()
        trace = []
        self.present_sol, self.present_dist = self.randompick() 
        while self.T >= self.min_T and time.time() - start < self.max_time:
            
            nextelem = list(self.present_sol)
            a = rd.randint(2, self.n - 1)
            b = rd.randint(0, self.n - a)
            nextelem[b:(a+b)] = reversed(nextelem[b:(a+b)])
            flag = self.criterion(nextelem)
            if flag == 1:
                trace.append([np.round((time.time()-start),2), self.present_dist])
            self.T *= self.alpha
            self.iter += 1
            self.best_list.append(self.present_dist)
            
        #print("Solution obtained: ", self.present_dist)
        #print("Best route: ", self.optimal_solution)
        
        return self.present_dist, self.optimal_solution, trace 
    
    def criterion(self, nextelem):
        
        flag = 0
        elem_dist = self.totdist(nextelem)
        if elem_dist < self.present_dist:
            self.present_dist, self.present_sol = elem_dist, nextelem
            if elem_dist < self.best_dist:
                self.best_dist, self.optimal_solution = elem_dist, nextelem
                flag = 1
        else:
            if rd.random() < self.isaccept(elem_dist):
                self.present_dist, self.present_sol = elem_dist, nextelem
        
        return flag
               
                
    def isaccept(self, elem_dist):
        
        return math.exp(-abs(elem_dist - self.present_dist)/self.T)
