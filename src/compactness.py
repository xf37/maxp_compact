#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:12:05 2020

@author: fengxin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:37:10 2020

@author: fengxin
"""

#import Polygon3
import csv
import math
import copy
import pysal as ps
import libpysal
import geopandas as gpd
from datetime import datetime
import numpy as np
import os
from copy import deepcopy
from scipy.sparse.csgraph import connected_components



# Default file path - can be overridden via environment variable
filePath = os.environ.get('MAXP_DATA_PATH', './data/')



class Polygon:   
    def _init_(self):
        self.data = [] # an array of points [x,y]
        self.area = 0.0
        self.centroidX = 0.0
        self.centroidY = 0.0
        self.id = 0
        self.inertia = 0.0
        self.shapeindex = 0.0   
        self.clockwise = True # True: closewise
        self.threshold = 0.0
        self.similarity = 0.0
        self.threshold = 0.0
        self.regionID = 0.0
        
    def __init__(self,points,id,similarity, threshold):
        self.data = points
        #print (points)
        self.id = id
        self.similarity = similarity
        self.threshold = threshold
        self.regionID = 0.0
        self.closewise = self.getClockwise(points)
        #print "Clockwise or not: ", self.closewise
        if not self.closewise:
            self.data = points[::-1]         
        self.getPolygonInertia()
        

    def getClockwise(self,points):
        area = 0.0
        for i in range(len(points)-1):
            pt2 = points[i+1]
            pt1 = points[i]
            area  += (pt2[0]-pt1[0])*(pt2[1]+pt1[1])/2
        return area > 0
            
        
    def distance(self,pointX, pointY):
        return math.sqrt((pointX[0]-pointY[0])**2+(pointX[1]-pointY[1])**2)
    
    # return neighbors unlabeled
    def getNeighbors(self, w, label):
        neighborPolys = deepcopy(w.neighbors[self.id])
        neighborSet_unlabeled = set(neighborPolys) 
        for neighborID in neighborPolys:
            if not label[neighborID] == 0:
                neighborSet_unlabeled.remove(neighborID) 
        return neighborSet_unlabeled 
                    
    
    #compute the moment of inertia of the ith polygon
    #Segment {poly[i],poly[i+1]}
    def calMomentOfInertia(self, i):
        
        poly = self.data
        
        xi = poly[i][0]
        xi_plus_1 = poly[i+1][0]
        yi = poly[i][1]
        yi_plus_1 = poly[i+1][1]
        # compute triangle
        areaTri = (xi_plus_1-xi)*(yi_plus_1-yi)*0.5
        #print (i,"areaTri", areaTri)

        IgTri = areaTri*((xi_plus_1-xi)**2+(yi_plus_1-yi)**2)/18
        x1g = (xi+2*xi_plus_1)/3
        y1g = (2*yi+yi_plus_1)/3
        
        #print (i,"IgTri", IgTri)
            
        # compute rectangle
        x2g = (xi+xi_plus_1)/2
        areaRec = yi*(xi_plus_1-xi)
        y2g = yi/2
        IgRec = areaRec*((xi_plus_1-xi)**2+yi**2)/12
        
        #print (i,"areaRec", areaRec)
        #print (i,"IgRec", IgRec)
         
        if areaTri==0.0 and areaRec == 0.0: #vertical
            returnshp =  [0.0,0.0,0.0,0.0]
        elif areaTri == 0.0:  #horizonal
            returnshp = [IgRec,areaRec,x2g,y2g]
        elif areaRec ==0.0: 
            returnshp = [IgTri,areaTri,x1g,y1g]
        else:
            returnshp = self.merge(
                    [IgTri,areaTri,x1g,y1g],[IgRec,areaRec,x2g,y2g])
        return returnshp
    
    
    
    def merge(self,shape1,shape2):
        area1 = shape1[1]
        area2 = shape2[1] 
        area = shape1[1] + shape2[1]
        #print (area, area1, area2)
        #print area
        rshape = []
        if area1 != 0.0 and area2 != 0.0 and area != 0.0:
            xg = (area1*shape1[2]+area2*shape2[2])/area
            yg = (area1*shape1[3]+area2*shape2[3])/area
            Ig = shape1[0] + shape2[0] + area1*(self.distance(
                    [shape1[2],shape1[3]],[xg,yg])**2)+area2*(
                    self.distance([shape2[2],shape2[3]],[xg,yg])**2)
            rshape = [Ig, area, xg, yg]
        elif area1 == 0.0:
            rshape = shape2
        elif area2 == 0.0:
            rshape = shape1
        else:
            rshape = [0.0,0.0,0.0,0.0]
        return rshape 
    
    def getPolygonInertia(self):
        poly = self.data
        shape = [0.0,0.0,0.0,0.0]
        for i in range(len(poly)-1):
            shape1= self.calMomentOfInertia(i)
            #print ("closewise?", self.closewise)
            shape = self.merge(shape,shape1)
            #print ("merge", i, shape[0], shape[1])
            
        self.inertia = shape[0]
        self.area = shape[1]
        self.centroidX = shape[2]
        self.centroidY = shape[3]
        self.shapeindex = 1-(self.area)**2/(2 * math.pi * self.inertia)

###############################################################################
        
class shpProc:
    def _init_(self):
        self.polygons ={}
        self.size = 0
        self.shpname = ""
        self.outputname = "TAZInertia.csv"

    
    def __init__(self, shpname, attrs_name, threshold_name, outputname="TAZInertia.csv", write_output=False):
        #print ("initializing instance")
        self.shpname = shpname
        self.outputname = outputname
        self.write_output = write_output
        self.polygons = {}
        polyDic, similarity,threshold = self.readSHP(shpname, attrs_name, threshold_name)
        self.initialize(polyDic, similarity, threshold)
    
    
    def initialize(self, polygons, similarity, threshold):
        #print("calculating moment of inertia for every polygon")
        time1 = datetime.now()
        for i in polygons:
            #print (i)
            polygon = Polygon(polygons[i],i,similarity[i],threshold[i]) # moment of inertia is calculated here
            #print (polygon.shapeindex)
            polygon.id = i
            self.polygons[i] = polygon

            #print (polygon)
        #print ("done.")
        self.size = len(self.polygons)
        time2 = datetime.now()
        #print ("Time Elapse: ", str(time2-time1))
        if self.write_output:
            global filePath
            os.makedirs(filePath, exist_ok=True)
            self.writeToFile(filePath + self.outputname)



    def writeToFile(self, output):
        with open(output, 'w') as csvfile:
            for i in self.polygons:
                csvfile.write("%s, %s, %s, %s, %s, %s, %s, %s\n" %(
                     self.polygons[i].id,
                     self.polygons[i].shapeindex, 
                     self.polygons[i].inertia,
                     self.polygons[i].area,
                     self.polygons[i].centroidX,
                     self.polygons[i].centroidY,
                     self.polygons[i].similarity,
                     self.polygons[i].threshold    # polygon Id starts from 1           
                     ))
            
        
    def readSHP(self, shpname, attrs_name, threshold_name):
        f = libpysal.io.open(shpname)
        #print ("initializing geoprocessors.. ")
        time1 = datetime.now()
        all_polygons = f.read()
        n_polygons = len(all_polygons)
        
        # initialization
        # read polygons
        polygons = {}
        #print ("start reading...")
        
        for index in range(n_polygons):
            partnum = 0
            # Count the number of points in the current polygon
            A_polygon = all_polygons[index].parts[0]
            partcount = len(A_polygon)
            points = []  # points array of each polygon
            
            while partnum < partcount:    
                pnt = A_polygon[partnum]
                points.append([pnt[0],pnt[1]])
                partnum += 1
                
            # id -1 means that polygon ID start from 0    
            polygons[all_polygons[index].id - 1] = points  
            
        # record aother attributes
        f_attribute = gpd.read_file(shpname)
        similarity = f_attribute[attrs_name]
        threshold = f_attribute[threshold_name]
        
        #print ("done.")
        time2 = datetime.now()
        #print ("Time Elapse: ", str(time2-time1))
        return polygons, similarity, threshold

 
###############################################################################

class Region:   
    def __init__(self):
        self.data = set() # a set of ID of polygons in this region
        self.area = 0.0
        self.seed = 0
        self.centroidX = 0.0
        self.centroidY = 0.0
        self.id = 0
        self.isEnclave = True
        self.inertia = 0.0         # value: inertia of the region
        self.shapeindex = 0.0  
        self.withinRegionHetero = 0.0   # within-region heterogeneity
        self.spatialAttrTotal = 0.0

        
    def __init__(self, Shp, seed, C, w=None, label=None):       
        self.data = set([seed]) # a set of ID of polygons in this region
        self.area = Shp[seed].area
        self.seed = Shp[seed].id        # the ID of seed polygon
        self.centroidX = Shp[seed].centroidX
        self.centroidY = Shp[seed].centroidY
        self.id = C
        self.isEnclave = True
        self.inertia = Shp[seed].inertia
        self.shapeindex = Shp[seed].shapeindex
        self.withinRegionHetero = 0.0   ## ??
        self.spatialAttrTotal = Shp[seed].threshold
        # record unlabeled neighbors
        if w != None and label != None:                    
            self.NeighborPolysID = Shp[seed].getNeighbors(w, label) # ID of neigboring polygons
        
    def distance(self,pointX, pointY):
        return math.sqrt((pointX[0]-pointY[0])**2+(pointX[1]-pointY[1])**2)
     
    def calculateWithinRegionHetero(self, distance_matrix):
        nv = np.array(list(self.data))
        regionDistance = distance_matrix[nv, :][:, nv].sum() / 2
        return regionDistance

    # check whether meet the requirement of the threshold
    def isRegion(self, threshold):
        if self.spatialAttrTotal >= threshold:
            self.isEnclave = False
            return True
        else:
            return False
        
    # Region Growth: combine one unit leading to best compactness
    # return the ID of the combined unit
    def selectUnit_growRegion(self, Shp, label, w, flag_rg, randomGrow): 
        #print (flag_rg == 1)              
        shape1 = [self.inertia,self.area,self.centroidX,self.centroidY]
        if flag_rg == 0:
            # add the first neighboring units into the region
            polysID = self.NeighborPolysID.pop()
            neighbor_p = Shp[polysID]
            shape2 = [neighbor_p.inertia, neighbor_p.area, neighbor_p.centroidX, neighbor_p.centroidY]
            rshape = []
            area1 = shape1[1]
            area2 = shape2[1]
            area = area1 + area2
            if shape1[1] != 0.0 and shape2[1] != 0.0:
                centroidX = (area1 * shape1[2] + area2 * shape2[2]) / area
                centroidY = (area1 * shape1[3] + area2 * shape2[3]) / area
                inertia = shape1[0] + shape2[0] + area1 * (self.distance([shape1[2],shape1[3]], [centroidX,centroidY])**2) + area2 * (self.distance([shape2[2],shape2[3]], [centroidX,centroidY])**2)
                rshape = [inertia, area, centroidX, centroidY]
            elif area1 == 0.0:
                rshape = shape2
            elif area2 == 0.0:
                rshape =shape1
            else:
                rshape = [0.0, 0.0, 0.0, 0.0] 
            shapeindex = 1-(area ** 2) / (2 * math.pi * inertia)
            
            #update region
            self.area = rshape[1]
            self.centroidX = rshape[2]
            self.centroidY = rshape[3]
            self.inertia = rshape[0]
            self.shapeindex = shapeindex
            self.spatialAttrTotal += neighbor_p.threshold
            self.data.add(neighbor_p.id)   # add unit in region's data
            self.NeighborPolysID = self.NeighborPolysID.union(neighbor_p.getNeighbors(w, label))
            return neighbor_p 
        
        elif flag_rg == 1:
            updateRegionList = []
            for polysID in self.NeighborPolysID:
                neighbor_p = Shp[polysID]
                shape2 = [neighbor_p.inertia, neighbor_p.area, neighbor_p.centroidX, neighbor_p.centroidY]
                rshape = []
                area1 = shape1[1]
                area2 = shape2[1]
                area = area1 + area2
                if shape1[1] != 0.0 and shape2[1] != 0.0:
                    centroidX = (area1 * shape1[2] + area2 * shape2[2]) / area
                    centroidY = (area1 * shape1[3] + area2 * shape2[3]) / area
                    inertia = shape1[0] + shape2[0] + area1 * (self.distance([shape1[2],shape1[3]], [centroidX,centroidY])**2) + area2 * (self.distance([shape2[2],shape2[3]], [centroidX,centroidY])**2)
                    rshape = [inertia, area, centroidX, centroidY]
                elif area1 == 0.0:
                    rshape = shape2
                elif area2 == 0.0:
                    rshape =shape1
                else:
                    rshape = [0.0, 0.0, 0.0, 0.0] 
                shapeindex = 1-(area ** 2) / (2 * math.pi * inertia)
                updateRegionList.append((shapeindex,rshape,neighbor_p))
            
            updateRegionList = sorted(updateRegionList, key =lambda tup: tup[0])
            top_num = min([len(updateRegionList), randomGrow])
            if top_num > 0:
                unit_index = np.random.randint(top_num)
                #print (unit_index)
                min_shapeindex = updateRegionList[unit_index][0]
                min_rshape = updateRegionList[unit_index][1]
                min_polygon = updateRegionList[unit_index][2]
                   
            # update region 
                self.area = min_rshape[1]
                self.centroidX = min_rshape[2]
                self.centroidY = min_rshape[3]
                self.inertia = min_rshape[0]
                self.shapeindex = min_shapeindex
                self.spatialAttrTotal += min_polygon.threshold
                self.data.add(min_polygon.id)   # add unit in region's data
                self.NeighborPolysID = self.NeighborPolysID.union(min_polygon.getNeighbors(w, label))
                self.NeighborPolysID.remove(min_polygon.id)
            return min_polygon 
        
        else:
            print("flag_rg should be either 0 or 1!")
    
    def enclaveAssign(self, enclave):
        shape1 = [self.inertia,self.area,self.centroidX,self.centroidY]
        shape2 = [enclave.inertia, enclave.area, enclave.centroidX, enclave.centroidY]
        rshape = []
        area1 = shape1[1]
        area2 = shape2[1]
        area = area1 + area2
        if shape1[1] != 0.0 and shape2[1] != 0.0:
            centroidX = (area1 * shape1[2] + area2 * shape2[2]) / area
            centroidY = (area1 * shape1[3] + area2 * shape2[3]) / area
            inertia = shape1[0] + shape2[0] + area1 * (self.distance([shape1[2],shape1[3]], [centroidX,centroidY])**2) + area2 * (self.distance([shape2[2],shape2[3]], [centroidX,centroidY])**2)
            rshape = [inertia, area, centroidX, centroidY]
        elif area1 == 0.0:
            rshape = shape2
        elif area2 == 0.0:
            rshape =shape1
        else:
            rshape = [0.0, 0.0, 0.0, 0.0] 
        shapeindex = 1-(area ** 2) / (2 * math.pi * inertia)
        DiffShapeindex = shapeindex - self.shapeindex
        return DiffShapeindex, rshape, shapeindex
        
        
    def remove_unit(self, unit):
        polygon_set = self.data
        if unit.id in polygon_set:
            shape1 = [self.inertia,self.area,self.centroidX,self.centroidY]
            shape2 = [unit.inertia, unit.area, unit.centroidX, unit.centroidY]
        
            #update values:
            rshape = []
            area1 = shape1[1]
            area2 = shape2[1]
            area = area1 - area2
            
            centroidX = (area1 * shape1[2] - area2 * shape2[2]) / area
            centroidY = (area1 * shape1[3] - area2 * shape2[3]) / area
            inertia = shape1[0] - shape2[0] - area * (self.distance([shape1[2],shape1[3]], [centroidX,centroidY])**2) - area2 * (self.distance([shape2[2],shape2[3]], [shape1[2],shape1[3]])**2)
            rshape = [inertia, area, centroidX, centroidY]
            shapeindex = 1-(self.area)**2/(2* math.pi *self.inertia)
            spatialAttrTotal = self.spatialAttrTotal - unit.threshold
            # wighted shapeindex difference before and after the remove
            shapeindexDiff = shapeindex * spatialAttrTotal  -  \
                             self.shapeindex * self.spatialAttrTotal
            return rshape, shapeindex, shapeindexDiff
     
        
    def combine_unit(self, unit):
        shape1 = [self.inertia,self.area,self.centroidX,self.centroidY]
        shape2 = [unit.inertia, unit.area, unit.centroidX, unit.centroidY]
        rshape = []
        area1 = shape1[1]
        area2 = shape2[1]
        area = area1 + area2
        if shape1[1] != 0.0 and shape2[1] != 0.0:
            centroidX = (area1 * shape1[2] + area2 * shape2[2]) / area
            centroidY = (area1 * shape1[3] + area2 * shape2[3]) / area
            inertia = shape1[0] + shape2[0] + area1 * (self.distance([shape1[2],shape1[3]], [centroidX,centroidY])**2) + area2 * (self.distance([shape2[2],shape2[3]], [centroidX,centroidY])**2)
            rshape = [inertia, area, centroidX, centroidY]
        elif area1 == 0.0:
            rshape = shape2
        elif area2 == 0.0:
            rshape =shape1
        else:
            rshape = [0.0, 0.0, 0.0, 0.0] 
        shapeindex = 1-(area ** 2) / (2 * math.pi * inertia)
        spatialAttrTotal = self.spatialAttrTotal + unit.threshold
        # wighted shapeindex difference before and after the remove
        shapeindexDiff = shapeindex * spatialAttrTotal  -  \
                         self.shapeindex * self.spatialAttrTotal
        return rshape, shapeindex, shapeindexDiff
        
###############################################################################
            
class Partition:    
    def __init__(self, n_polygons):
        self.label = [0] * n_polygons # a set of labels for polygons
        self.data = []
        self.id = 0
        self.p = 0  # number of regions
        self.shapeindex = 0.0   # mean of shapeindex for regions
        self.totalwithinRegionHetero = 0.0
    
    def calculateWithinRegionHeteroTotal(self, distance_matrix):
        totalWithinRegionHetero = 0.0
        for region in self.data:
            regionDistance = region.withinRegionHetero
            totalWithinRegionHetero += regionDistance
        return totalWithinRegionHetero
    
    def calculateShapeIndex(self):
        sum_shapeindex = 0
        weight = 0    # attribute total 
        for region in self.data:
            sum_shapeindex += region.shapeindex * region.spatialAttrTotal
            weight += region.spatialAttrTotal
        meanWeightedShapeindex = sum_shapeindex/weight
        return meanWeightedShapeindex
            
    def pickMoveArea(self, Shp, w, distance_matrix, region_threshold):
        potentialAreas = []
        for ARegion in self.data:
            rla = np.array(list(ARegion.data))    # polygons belong to region k
            v = ARegion.spatialAttrTotal
            rasa = []
            for i in list(ARegion.data):
                rasa.append(Shp[i].threshold)
            rasa = np.array(rasa)
            # condition 1: left spatial attribute value > threshold
            lostSA = v - rasa
            pas_indices = np.where(lostSA > region_threshold)[0]
            # condition 2: left areas are connected
            if pas_indices.size > 0:
                for pasi in pas_indices:
                    leftAreas = np.delete(rla, pasi)
                    ws = w.sparse
                    cc = connected_components(ws[leftAreas, :][:, leftAreas])
                    if cc[0] == 1:
                        potentialAreas.append(rla[pasi])
            else:
                continue
        #print (potentialAreas)
        return potentialAreas
        
    # check whether move polygon poa from donorRegion
    def checkMove(self, Shp, poa, w, distance_matrix, region_threshold):
        poaNeighbor = w.neighbors[poa]    # poa is polygon ID
        donorRegionID = self.label[poa]
        
        rm = np.array(list(self.data[donorRegionID-1].data))
        lostDistance = distance_matrix[poa, rm].sum()
        # lostCompactness: shapeindex change because of removing
        # -1 : regionID start from 1, but partition.data is a list
        lostRshape,lostShapeIndex, lostShapeIndexDiff = self.data[donorRegionID-1].remove_unit(Shp[poa])
        potentialMove = {}
        
        minAddedDistance = np.Inf
        NeighborIDset = []
        # check each neighbor
        for poan in poaNeighbor:
            recipientRegionID = self.label[poan]          
            if recipientRegionID not in NeighborIDset and \
               donorRegionID != recipientRegionID:
                NeighborIDset.append(recipientRegionID)
                # within-class hetero
                rm = np.array(list(self.data[recipientRegionID-1].data))
                addedDistance = distance_matrix[poa, rm].sum()
                # compactness
                # addedCompactness: shapeindex change because of combining
                addedRshape,addedShapeIndex, addedShapeIndexDiff = \
                        self.data[recipientRegionID-1].combine_unit(Shp[poa])

                
                potentialMove[recipientRegionID] = [lostDistance, addedDistance,\
                    lostShapeIndexDiff, addedShapeIndexDiff, \
                    lostRshape, addedRshape, lostShapeIndex, addedShapeIndex]
            #print (potentialMove)
        return potentialMove       
                
                
    def performSA(self, Shp, w, distance_matrix, threshold, 
                  alpha, tabuLength, max_no_move):
        t = 1   # temperature
        ni_move_ct = 0     # nonimproving moves count
        make_move_flag = False
        tabuList = []
        potentialAreas = []    # potential units satisfied moving requirements
    
        #labels = deepcopy(initLabels)
        #regionLists = deepcopy(initRegionList)
        #regionSpatialAttrs = deepcopy(initRegionSpatialAttr)
    
        while ni_move_ct <= max_no_move and t > 0.1:
            if len(potentialAreas) == 0:
                # identify PU
                potentialAreas = self.pickMoveArea(Shp, w, distance_matrix, threshold)
    
            if len(potentialAreas) == 0:
                break
            
            # randomly select a unit
            poa = potentialAreas[np.random.randint(len(potentialAreas))]

            #  get potential move for the selected unit
            potentialMove = self.checkMove(Shp, poa, w, distance_matrix, threshold)
    
            if len(potentialMove) == 0:
                potentialAreas.remove(poa)
                continue
                
            group1 = []   # improving both compactness & hetero
            group2 = []   # improving compactness 
            group3 = []   # improving hetero
            group4 = []   # nonimproving both compactness & hetero
            
            min_diffCompact_regionID = 0
            min_diffCompact = np.inf
            donorRegion = self.label[poa]
            for recipientRegion in potentialMove:
                # calculate difference of distance
                diffDistance = potentialMove[recipientRegion][0] - \
                                potentialMove[recipientRegion][1]
                # calculate changing of compactness (no weight ## may revise in the future)
                diffCompactness = potentialMove[recipientRegion][2]  + \
                                potentialMove[recipientRegion][3]
                if diffCompactness < 0:
                    if diffDistance > 0:
                        group1.append(recipientRegion)
                    else:
                        group2.append(recipientRegion)
                else:
                    if diffDistance > 0:
                        group3.append(recipientRegion)
                    else:
                        group4.append(recipientRegion)
                if diffCompactness < min_diffCompact:
                    min_diffCompact = diffCompactness
                    min_diffCompact_regionID = recipientRegion
                
            #Case 0: best compactness 
            randomRecipient = min_diffCompact_regionID
            if  min_diffCompact < 0:                   
                make_move_flag = True
                # add the reserve move to TabuList
                reserve_mv = (poa, randomRecipient, donorRegion)
                if reserve_mv not in tabuList:
                    # ?? reach the maximum tabulength, then delete one item??
                    if len(tabuList) == tabuLength:
                        tabuList.pop(0)
                    tabuList.append(reserve_mv)
                ni_move_ct = 0      # reset to 0      
            else:
                # print ("group3_4 is:", group3_4)
                ni_move_ct += 1
                #print (ni_move_ct)
                md_value = min_diffCompact / t
                if md_value <= 1e-20 and md_value >= -1e-20:
                    md_value = 0
                prob = np.exp(- md_value)
                #print (prob)
                pot_mv = (poa, donorRegion, randomRecipient)
                if prob > np.random.random_sample() and pot_mv not in tabuList:
                    make_move_flag = True
                else:
                    make_move_flag = False


            # update data
            potentialAreas.remove(poa)
            if make_move_flag:
                
                # for partition:
                self.label[poa] =  randomRecipient

                #self.shapeindex += potentialMove[randomRecipient][2] + \
                #                potentialMove[randomRecipient][3]
                #self.totalwithinRegionHetero += potentialMove[randomRecipient][0] - \
                #                potentialMove[randomRecipient][1]
                             
                # for regions:
                # for donor region:
                self.data[donorRegion-1].data.remove(poa)
                self.data[donorRegion-1].spatialAttrTotal -= Shp[poa].threshold
                self.data[donorRegion-1].area = potentialMove[randomRecipient][4][1]
                self.data[donorRegion-1].centroidX = potentialMove[randomRecipient][4][2]
                self.data[donorRegion-1].centroidY = potentialMove[randomRecipient][4][3]
                self.data[donorRegion-1].inertia = potentialMove[randomRecipient][4][0]
                self.data[donorRegion-1].shapeindex = potentialMove[randomRecipient][6]             
                self.data[donorRegion-1].withinRegionHetero\
                                        -= potentialMove[randomRecipient][0]
                
                # for recipient region:                       
                self.data[randomRecipient-1].data.add(poa)           
                self.data[randomRecipient-1].spatialAttrTotal += Shp[poa].threshold
                self.data[randomRecipient-1].area = potentialMove[randomRecipient][5][1]
                self.data[randomRecipient-1].centroidX = potentialMove[randomRecipient][5][2]
                self.data[randomRecipient-1].centroidY = potentialMove[randomRecipient][5][3]
                self.data[randomRecipient-1].inertia = potentialMove[randomRecipient][5][0]
                self.data[randomRecipient-1].shapeindex = potentialMove[randomRecipient][7]
                self.data[randomRecipient-1].withinRegionHetero \
                                         += potentialMove[randomRecipient][1]

                
                impactedAreas = []
                for pa in potentialAreas:
                    if self.label[pa] == randomRecipient or self.label[pa] == donorRegion:
                        impactedAreas.append(pa)
                for pa in impactedAreas:
                    potentialAreas.remove(pa)
    
            t = t * alpha
    

               
###############################################################################





        
        