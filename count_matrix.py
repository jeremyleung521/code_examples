#!/usr/bin/env python
# count_matrix.py
#
# Python script that calculates transition count matrix from a file called all_eligible.dat and cross
# reference with that in cnumvtime.dat generated from CPPTRAJ clustering.
# 
# Resulting matrices (raw and normalized) are outputted into a numpy array 
# in transition_matrix.npy and transition_matrix.dat.
# Use the .npy for easy reloading into python,  using the following code:
# with open("transition_matrix.npy", 'rb') as f:
#     a = numpy.load(f) # load raw matrix
#     b = numpy.load(f) # load normalized matrix
#
# Make sure to replace all "not looked at" and strings in your all_eligible.dat to integers. 'all_eligible.dat' is
# a concatenated version of all eligible_states.dat from each independent trial. 
# It is possible the transition matrix to contain a row of zeros, corresponding to sources or sinks. This will require 
# inspection of your clustering.
#
# Written by Jeremy Leung
#

import numpy
import sys
import os

###Things to Change###
c = 10 # number of clusters. Change as necessary.
m = 5 # number of files in multi. Change as necessary.
lag = 1 #Lag time/number of iterations to look back. Change as necessary. (Default: 1; one iteration back)

###I/O (Things to Change)##
it, seg, j, pid  = numpy.loadtxt("all_eligible.dat", skiprows=1, usecols=(0,1,4,5), unpack=True, dtype=int) # file connecting frame (column) with iter/seg/parent/file info
h = numpy.loadtxt("all_eligible.dat", skiprows=1, usecols=3, unpack=True, dtype=float) # file connecting frame (column) with weight info
k = numpy.loadtxt("cnumvtime.dat", skiprows=1, usecols=1, unpack=True, dtype=int) # file with cluster assignments

##I/O Management##
it = it.tolist() # turning the following readouts to lists for the index() command. Hopefully quicker than np.where()
j = j.tolist()
seg = seg.tolist()
pid = pid.tolist()


####CODE####
count = [] #initialize counting matrix to 0s
for i in range(0,c):
    count.append([0.]*c)
#iter = 0. # Variable for calcuating normalization constant

fid = {}
for i in range(1,m+1): # create a dictionary of indices for the first row of each file
    fid[i] = j.index(i)

for i in range(0,len(h)): # going through rows in bounds_states.dat
    if it[i] > lag:
        a = it.index((it[i] - lag),fid[j[i]]) #find first index with iteration number - lag (with same file number)
        try:  
            b = seg.index((pid[i]),a) # find index of parent seg, searching from iteration number -1
            x = k[b] # starting cluster of parent traj
            y = k[i] # ending cluster of current traj
            count[x][y] += h[i] # add weight to count matrix
#            iter += h[i] # add weight for normalization later on
        except ValueError:
            with open("count_exceptions.dat", "a") as file:
                file.write("iter:" + str(it[i]).ljust(5) + "\tseg:" + str(seg[i]) + "\tfile:" + str(j[i]) + "\n")
#    else: # incomplete code for negative parent_ids from istates
#        pid[i] = pid[i] / 15

#with open("weight_count.dat",'a') as g: # writing all the info into a file called "weight_count.dat"
#    g.write("cluster" + "\t" + "weight".format() + "\t\t\t" + "normalized" + "\n")
#    for i in range(0,c):
#        g.write(str(i) + "\t" + str(d[i])+ "\t" + str(d[i]/iter) + "\n")

#normcount = numpy.asarray(count / iter)
count = numpy.asarray(count)

sum_of_rows = count.sum(axis=1)
normcount = count / sum_of_rows[:, numpy.newaxis]

with open("transition_matrix.npy",'wb') as file:
    numpy.save(file, count)
    numpy.save(file, normcount)

with open("transition_matrix.dat", "w") as file:
    file.write(str(count))
    file.write("\n" + "normalized:" + "\n")
    file.write(str(normcount))
