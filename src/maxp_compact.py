#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:47:00 2020

@author: fengxin
"""

import numpy
import libpysal
import geopandas as gpd
import pandas as pd
import os
import sys
from datetime import datetime
#import math
from compactness import Polygon, shpProc, Region, Partition

###############################
#from BaseClass import BaseSpOptHeuristicSolver
#from base import (w_to_g, move_ok, ok_moves, region_neighbors, _centroid,
#                   _closest, _seeds, is_neighbor)

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import geopandas as gp
import numpy as np
from copy import deepcopy
from scipy.sparse.csgraph import connected_components

flag_rg = int(sys.argv[1])    # whether to consider compactness in region grow
flag_ea = int(sys.argv[2])     # whether to consider compactness in enclave assign
flag_ls = int(sys.argv[3])     # whether to consider compactness in local search
randomGrow = int(sys.argv[4])
randomAssign = int(sys.argv[5])
threshold = int(sys.argv[6])

# local test
#flag_rg = 1     # whether to consider compactness in region grow
#flag_ea = 1    # whether to consider compactness in enclave assign
#flag_ls = 1   # whether to consider compactness in local search
#randomGrow = 2
#randomAssign = 2

ITERCONSTRUCT=1000  #??
ITERSA = 100    #??
max_totoalhetero = np.inf  # total within-class heterogeneity

###############################
## example data : n100.shp
#pth = '/rhome/xinfeng/bigdata/compact_maxp/data/n100.shp'
#file_path = '/rhome/xinfeng/bigdata/compact_maxp/result/'
#outputname =  "n100_" + str(flag_rg) + "_" + str(flag_ea) + "_" + str(flag_ls)\
#             + "rg" + str(randomGrow) + "ra" + str(randomAssign) + ".csv"


#local test
#pth = '/Users/xinfeng/Dropbox/Work/UCRiverside/RIDIR/data/n100.shp'
#file_path = '/Users/xinfeng/Dropbox/Work/UCRiverside/RIDIR/result/'
#outputname =  "C_n100_" + str(flag_rg) + "_" + str(flag_ea) + "_" + str(flag_ls)\
#             + "rg" + str(randomGrow) + "ra" + str(randomAssign) + ".csv"

#f = gpd.read_file(pth)
#threshold_name = 'Uniform2'
#attrs_name_str = 'SAR1'
#attrs_name = [attrs_name_str]

#w = libpysal.weights.Queen.from_dataframe(f)
#threshold = 50
  
## example data :TAZ
pth = '/home/sfeng/compact_maxp/data/taz/TAZ_MHI_POP.shp'
file_path = '/home/sfeng/compact_maxp/result/'
outputname =  "TAZ_MHI_POP_" + str(flag_rg) + "_" + str(flag_ea) + "_" + str(flag_ls)\
            + "rg" + str(randomGrow) + "ra" + str(randomAssign) + "_" + "th" + str(threshold) + ".csv"


#local test           
# pth = '/Users/xinfeng/Dropbox/Work/UCRiverside/RIDIR/data/5counties/TAZ_MHI_POP.shp'
# file_path = '/Users/xinfeng/Dropbox/Work/UCRiverside/RIDIR/result/taz/'
# outputname =  "TAZ_MHI_POP_" + str(flag_rg) + "_" + str(flag_ea) + "_" + str(flag_ls)\
#              + "rg" + str(randomGrow) + "ra" + str(randomAssign) + "_" + str(threshold) + ".csv"


f = gpd.read_file(pth)
threshold_name = 'POP18'
attrs_name_str = 'MHI2016'
attrs_name = [attrs_name_str]

# weight calculation
w = libpysal.weights.Queen.from_dataframe(f)



# distance calculation
attr = f[attrs_name].values
distance_matrix = squareform(pdist(attr, metric='cityblock'))

# other parameters (for construction)
max_iterations_construction=ITERCONSTRUCT
max_iterations_sa=ITERSA
verbose=False

# other parameters (for local search)
alpha = 0.998
tabuLength = 10
max_no_move = 100
#max_no_move = 1
#print ("max_no_move is :", max_no_move)
best_obj_value = 1
best_partition = None
best_label = None
best_fn = None

###############################################################################
# Shape Processing

maxp_time1 = datetime.now()
Shp = shpProc(pth, attrs_name_str, threshold_name).polygons
n_polygons = len(Shp)   # n: number of polygons

###############################################################################
# Construction: region growth + enclave assignment

max_p = 0
partitions_list = []   # Partitions leading to max p
arr = np.arange(0, n_polygons) 
partition_id = 0
time_ea = 0


for iteration in range(max_iterations_construction):
    if iteration % 5000 == 0:
        print (iteration)
    #labels = [0] * n_polygons
    C = 0       # index of regions 
    enclaveList = []
    np.random.shuffle(arr)   # Modify a sequence in-place by shuffling its contents.
    APartition = Partition(n_polygons) # New a partition
    
    for index in range(0, n_polygons):
        
        P = arr[index]
        #print ("P = ", P)
        
        # check if P is already labeled 
        if not (APartition.label[P] == 0):
            continue
        
        # grow region from seed P
        C += 1        
        APartition.label[P] = C
        #print ("Regrion: ", C)
        ARegion = Region(Shp, P, C, w, APartition.label) #  new a region
        #print (ARegion.data)
        
        while len(ARegion.NeighborPolysID) > 0:  # keep growing until no unlabeled neighbors
            if ARegion.isRegion(threshold):   # check whetaher meet the requirement
                break
            # 1) combine a unit to a region
            # 2) get back the id of the unit and label it
            #print (flag_rg)
            combined_unit = ARegion.selectUnit_growRegion(Shp, APartition.label, w, flag_rg,randomGrow) 
            APartition.label[combined_unit.id] = C
        
        #print (ARegion.data)    
        # record the enclave information
        if not ARegion.isRegion(threshold):
            C -= 1
            #print ("Enclave")
            enclaveList = enclaveList + list(ARegion.data)
        else:
            #print ("it is a region", C)
            ARegion.withinRegionHetero = ARegion.calculateWithinRegionHetero(distance_matrix)
            #print ("withinRegionHetero for region", ARegion.id, "is ", ARegion.withinRegionHetero) 
            APartition.p += 1
            APartition.data.append(ARegion)

            
    # update max_p  
    if APartition.p < max_p:
        continue
    else: 
        maxp_time2 = datetime.now()
        max_p = APartition.p
        #print ("find a new max_p:", max_p)
           
        if flag_ea == 1:       
                 # assign enclave 
            enclave_index = 0
            #print ("enclaveList: ", enclaveList)
            while len(enclaveList) > 0:
                regionList = []
                updatedRegionList = []
                min_rshape = []
                min_DiffShapeindex = 1.0
                min_region = None
                AEnclave = enclaveList[enclave_index]
                ecNeighbors = w.neighbors[AEnclave]
                for ecn in ecNeighbors:
                    if ecn in enclaveList:
                        continue
                    else:
                        #regionList.append(APartition.data[APartition.label[ecn]])
                        if APartition.label[ecn] not in regionList:
                            regionList.append(APartition.label[ecn])
                        
                        
                if len(regionList) == 0:
                    enclave_index += 1
                else:
                    for regionID in regionList:
                        region = APartition.data[regionID -1]
                        DiffShapeindex, rshape, shapeindex = region.enclaveAssign(Shp[AEnclave])
                        #print ("shapeindex of enclave", ecn, "for region", regionID, "is ", shapeindex)
                        updatedRegionList.append((regionID, DiffShapeindex,rshape,shapeindex))
                            
                    updatedRegionList = sorted(updatedRegionList, key =lambda tup: tup[1]) 
                    top_n = min([len(updatedRegionList), randomAssign])
                    if top_n > 0:
                        unit_index = np.random.randint(top_n)
                     
                    # print ("enclave", AEnclave, "is assigned to region", min_region)
                    min_region = updatedRegionList[unit_index][0]
                    min_rshape = updatedRegionList[unit_index][2]
                    APartition.label[AEnclave] = min_region
                    updateRegion = APartition.data[min_region -1]
                    updateRegion.data.add(AEnclave)
                    updateRegion.area = min_rshape[1]
                    updateRegion.centroidX = min_rshape[2]
                    updateRegion.centroidY = min_rshape[3]
                    updateRegion.inertia = min_rshape[0]
                    updateRegion.shapeindex = updatedRegionList[unit_index][3]
                    updateRegion.spatialAttrTotal += Shp[AEnclave].threshold
                    updateRegion.withinRegionHetero = \
                    updateRegion.calculateWithinRegionHetero(distance_matrix)
                    # print ("withinRegionHetero for region", updateRegion.id, "is ", updateRegion.withinRegionHetero) 
                    del enclaveList[enclave_index]
                    enclave_index = 0
            
        elif flag_ea == 0:
        # assign enclave 
            enclave_index = 0
            #print ("enclaveList: ", enclaveList)
            while len(enclaveList) > 0:
                regionList = []
                min_rshape = []
                min_DiffShapeindex = 1.0
                min_region = None
                AEnclave = enclaveList[enclave_index]
                ecNeighbors = w.neighbors[AEnclave]
                for ecn in ecNeighbors:
                    if ecn in enclaveList:
                        continue
                    else:
                        #regionList.append(APartition.data[APartition.label[ecn]])
                        if APartition.label[ecn] not in regionList:
                            regionList.append(APartition.label[ecn])
                
                if len(regionList) == 0:
                    enclave_index += 1
                else:
                    regionindex = np.random.randint(len(regionList))
                    regionID = regionList[regionindex]
                    region = APartition.data[regionID -1]
                    DiffShapeindex, rshape, shapeindex = region.enclaveAssign(Shp[AEnclave])
                    
                    APartition.label[AEnclave] = regionID
                    updateRegion = APartition.data[regionID -1]
                    updateRegion.data.add(AEnclave)
                    updateRegion.area = rshape[1]
                    updateRegion.centroidX = rshape[2]
                    updateRegion.centroidY = rshape[3]
                    updateRegion.inertia = rshape[0]
                    updateRegion.shapeindex = shapeindex
                    updateRegion.spatialAttrTotal += Shp[AEnclave].threshold
                    updateRegion.withinRegionHetero = \
                    updateRegion.calculateWithinRegionHetero(distance_matrix)
                    del enclaveList[enclave_index]
                    enclave_index = 0
                    
                    
        
        # get partition shapeindex
        APartition.shapeindex = APartition.calculateShapeIndex()
        # get partition withinRegionHetero
        APartition.totalwithinRegionHetero = \
        APartition.calculateWithinRegionHeteroTotal(distance_matrix)
        # update partition's id
        APartition.id = partition_id
        
        maxp_time3 = datetime.now() 
        time_ea_1 = pd.to_timedelta((maxp_time3 - maxp_time2)) / pd.offsets.Second(1)          
        time_ea = time_ea + time_ea_1 
        
        partitions_list.append(APartition)  
        partition_id += 1
        
maxp_time4 = datetime.now()

###############################################################################
time_rg = pd.to_timedelta((maxp_time4-maxp_time1)) / pd.offsets.Second(1)  
time_rg_per = (time_rg-time_ea)/ITERCONSTRUCT
time_ea_per = time_ea / len(partitions_list)
time_instr_total =  time_rg
    
if verbose == True:
    print ("Total time elapse: ", str(maxp_time4-maxp_time1))  
    print (time_rg_per)
    print (time_ea_per)


n_maxp_partition = 0
minPartitionShape = 1.0
maxp_partition = []
for i in range(len(partitions_list)):
    if partitions_list[i].p == max_p:
        n_maxp_partition += 1
        maxp_partition.append(partitions_list[i])
        if partitions_list[i].shapeindex < minPartitionShape:
            bestPartition_before_ls = partitions_list[i]
            minPartitionShape = partitions_list[i].shapeindex
            minPartitionID = i

totalInRegionHetero = partitions_list[minPartitionID].totalwithinRegionHetero

if verbose == True:
    print ("the maximum p value found is", max_p, ",", 
           "the number of partitions for maxp is ", n_maxp_partition, ",", 
           "the best(minimum) shapeindex value found is",minPartitionShape, ",", 
           "its corresponding within region distance is", totalInRegionHetero, ",", 
           "its corresponding partition is", minPartitionID)


#selected = f[f.MYID == 175]
#selected.plot()

feasible_label = deepcopy(partitions_list[minPartitionID].label)
f["maxp"] = feasible_label
f.plot(column="maxp", categorical=True, figsize=(12,8), cmap='plasma') 


###############################################################################
# Local search 
maxp_time5 = datetime.now()
if flag_ls == 1:
     for partitioniter in range(len(partitions_list)):
        # for each partition, try max_iterations_sa times
        if partitions_list[partitioniter].p == max_p:
            for saiter in range(max_iterations_sa):
                #print (saiter)
                APartition = deepcopy(partitions_list[partitioniter])
                # stimulated annualing runs here
                APartition.performSA(Shp, w, distance_matrix, threshold, 
                          alpha, tabuLength, max_no_move)
                
                # update total WithinRegionDistance & shapeIndex
                APartition.totalwithinRegionHetero = \
                        APartition.calculateWithinRegionHeteroTotal(distance_matrix)
                APartition.shapeindex = APartition.calculateShapeIndex()
                
                # Case 1: compactness prior to hetero 
                if APartition.shapeindex < best_obj_value:
                    best_partition = APartition
                    best_obj_value = APartition.shapeindex 
                    # print (best_obj_value)
                    # print (best_partition.label)
                    # print (APartition.totalwithinRegionHetero)
if flag_ls == 0:
    best_partition = bestPartition_before_ls


                
maxp_time6 = datetime.now()
###############################################################################
maxShapeindex = 0
for region in best_partition.data:
    if region.shapeindex > maxShapeindex:
        maxShapeindex = region.shapeindex
        


time_ls = (maxp_time6-maxp_time5)/n_maxp_partition
time_ls_per = pd.to_timedelta(time_ls) / pd.offsets.Second(1) 
time_ls_total = pd.to_timedelta(maxp_time6-maxp_time5) / pd.offsets.Second(1) 

if verbose == True:
    print ("Total time elapse: ", str(maxp_time6-maxp_time5))  
    print ("the maximum p value found is", best_partition.p, ",", 
           "the best(minimum) shapeindex value found is",best_partition.shapeindex, ",", 
           "its corresponding within region distance is", \
           best_partition.totalwithinRegionHetero, ",", 
           "its corresponding maximum shapeindex of regions is", \
           maxShapeindex, ",",
           "its corresponding partition is", best_partition.id)

#selected = f[f.MYID == 175]
#selected.plot()

#h = 1 + math.floor(math.log(best_partition.shapeindex))
#obj_val = (-1 * best_partition.p) * math.pow( 10, h)+ best_partition.shapeindex
#print (obj_val, best_partition.p, best_partition.shapeindex)

#f["maxp_new"] = partitions_list[best_partition.id].label
#f["maxp_new"] = best_partition.label
#f.plot(column="maxp_new", categorical=True, figsize=(12,8), cmap='Spectral')        
#f.to_file(file_path + outputname + ".shp")
###############################################################################
# # write list to a file
# with open(file_path + outputname, 'w') as filehandle:
#             for label in maxp_partition:
#                 filehandle.write("%s\n" %label)

# # read the file to a list        
# with open(file_path + outputname, 'r') as filehandle:
#     maxp_partition = filehandle.readlines()
    
with open(file_path + outputname, 'a') as csvfile:
        csvfile.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(
             max_p,
             n_maxp_partition, 
             minPartitionShape,
             totalInRegionHetero,
             time_rg_per,
             time_ea_per,
             time_instr_total,
             best_partition.p,
             best_partition.shapeindex,
             best_partition.totalwithinRegionHetero,
             maxShapeindex,
             time_ls_per,
             time_ls_total
             ))
