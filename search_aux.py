#!/usr/bin/env python
# search_aux.py
#
# Python script that uses the KDTree algorithm to search for the nearest neightbor to a target 2D coordinate and outputs 
# the segment and iteration numbers.
#
# Written by Jeremy Leung
#

import numpy
import h5py
from scipy.spatial import KDTree
import sys

f = h5py.File("wsh2054_p2_west.h5", 'r')

target = [5,10.5] # This is the target value you want to look for 

# phase 1: finding iteration number
array1 = []
array2 = []
for i in range(1,1001): # change indices to number of iteration
  i = str(i)
  iteration = "iter_" + str(numpy.char.zfill(i,8))
  r1 = f['iterations'][iteration]['auxdata']['RMS_Helix2'][:,-1] # These are the auxillary coordinates you're looking for
  r2 = f['iterations'][iteration]['auxdata']['RMS_Helix3'][:,-1] # These are the auxillary coordinates you're looking for
  small_array = []
  for j in range(0,len(r1)):
    small_array.append([r1[j],r2[j]])
  tree = KDTree(small_array)
  dd, ii = tree.query(target,k=1) # Outputs are distance from neighbour (dd) and indices of output (ii)
  array1.append(dd) 
  array2.append(ii)
minimum = numpy.argmin(array1)
iter_num = str(minimum+1)

# phase 2: finding seg number
wheretolook = "iter_" + str(numpy.char.zfill(iter_num,8))
r1 = f['iterations'][wheretolook]['auxdata']['RMS_Helix2'][:,-1] # These are the auxillary coordinates you're looking for
r2 = f['iterations'][wheretolook]['auxdata']['RMS_Helix3'][:,-1] # These are the auxillary coordinates you're looking for
small_array2 = []
for j in range(0,len(r1)):
  small_array2.append([r1[j],r2[j]])
tree2 = KDTree(small_array2)
d2, i2 = tree2.query(target,k=1)  
seg_num = str(i2)
print("go to iter " + str(iter_num) + ", " + "and seg " + str(seg_num))
