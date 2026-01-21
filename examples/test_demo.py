#!/usr/bin/env python3
"""
Test script for max-p regionalization with compactness constraints.
Uses libpysal's Mexico sample dataset.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy

from compactness import Polygon, shpProc, Region, Partition

def main():
    print("=" * 60)
    print("Max-p Regionalization with Compactness Constraints - Test")
    print("=" * 60)

    # 1. Load sample data
    print("\n[1] Loading Mexico sample data from libpysal...")
    mexico = libpysal.examples.load_example('mexico')
    mexico_shp = mexico.get_path('mexicojoin.shp')
    gdf = gpd.read_file(mexico_shp)
    print(f"    Loaded {len(gdf)} polygons")
    print(f"    Columns: {list(gdf.columns)}")

    # 2. Set parameters
    print("\n[2] Setting parameters...")
    flag_rg = 1        # Enable compactness in region growth
    flag_ea = 1        # Enable compactness in enclave assignment
    randomGrow = 2
    randomAssign = 2
    threshold = 30000  # Higher threshold = fewer regions (easier to verify visually)
    ITERCONSTRUCT = 50 # Reduced for faster testing

    threshold_name = 'PCGDP1940'
    attrs_name_str = 'PCGDP1950'
    attrs_name = [attrs_name_str]

    print(f"    Threshold: {threshold}")
    print(f"    Iterations: {ITERCONSTRUCT}")

    # 3. Prepare data
    print("\n[3] Preparing spatial weights and distance matrix...")
    w = libpysal.weights.Queen.from_dataframe(gdf)
    print(f"    Spatial weights: {w.n} observations")

    attr = gdf[attrs_name].values
    distance_matrix = squareform(pdist(attr, metric='cityblock'))
    print(f"    Distance matrix: {distance_matrix.shape}")

    # 4. Process shapes
    print("\n[4] Processing shapes (calculating moment of inertia)...")
    start_time = datetime.now()
    Shp = shpProc(mexico_shp, attrs_name_str, threshold_name).polygons
    n_polygons = len(Shp)
    print(f"    Processed {n_polygons} polygons")
    print(f"    Time: {datetime.now() - start_time}")

    # 5. Run construction phase
    print("\n[5] Running construction phase...")
    maxp_time1 = datetime.now()

    max_p = 0
    partitions_list = []
    arr = np.arange(0, n_polygons)
    partition_id = 0

    for iteration in range(ITERCONSTRUCT):
        if iteration % 10 == 0:
            print(f"    Iteration {iteration}/{ITERCONSTRUCT}")

        C = 0
        enclaveList = []
        np.random.shuffle(arr)
        APartition = Partition(n_polygons)

        for index in range(n_polygons):
            P = arr[index]

            if APartition.label[P] != 0:
                continue

            C += 1
            APartition.label[P] = C
            ARegion = Region(Shp, P, C, w, APartition.label)

            while len(ARegion.NeighborPolysID) > 0:
                if ARegion.isRegion(threshold):
                    break
                combined_unit = ARegion.selectUnit_growRegion(Shp, APartition.label, w, flag_rg, randomGrow)
                APartition.label[combined_unit.id] = C

            if not ARegion.isRegion(threshold):
                C -= 1
                enclaveList = enclaveList + list(ARegion.data)
            else:
                ARegion.withinRegionHetero = ARegion.calculateWithinRegionHetero(distance_matrix)
                APartition.p += 1
                APartition.data.append(ARegion)

        if APartition.p >= max_p:
            max_p = APartition.p

            # Assign enclaves
            enclave_index = 0
            while len(enclaveList) > 0:
                regionList = []
                AEnclave = enclaveList[enclave_index]
                ecNeighbors = w.neighbors[AEnclave]

                for ecn in ecNeighbors:
                    if ecn not in enclaveList and APartition.label[ecn] not in regionList:
                        regionList.append(APartition.label[ecn])

                if len(regionList) == 0:
                    enclave_index += 1
                else:
                    if flag_ea == 1:
                        updatedRegionList = []
                        for regionID in regionList:
                            region = APartition.data[regionID - 1]
                            DiffShapeindex, rshape, shapeindex = region.enclaveAssign(Shp[AEnclave])
                            updatedRegionList.append((regionID, DiffShapeindex, rshape, shapeindex))

                        updatedRegionList = sorted(updatedRegionList, key=lambda tup: tup[1])
                        top_n = min(len(updatedRegionList), randomAssign)
                        unit_index = np.random.randint(top_n) if top_n > 0 else 0

                        min_region = updatedRegionList[unit_index][0]
                        min_rshape = updatedRegionList[unit_index][2]
                    else:
                        regionindex = np.random.randint(len(regionList))
                        min_region = regionList[regionindex]
                        region = APartition.data[min_region - 1]
                        _, min_rshape, _ = region.enclaveAssign(Shp[AEnclave])

                    APartition.label[AEnclave] = min_region
                    updateRegion = APartition.data[min_region - 1]
                    updateRegion.data.add(AEnclave)
                    updateRegion.area = min_rshape[1]
                    updateRegion.centroidX = min_rshape[2]
                    updateRegion.centroidY = min_rshape[3]
                    updateRegion.inertia = min_rshape[0]
                    updateRegion.spatialAttrTotal += Shp[AEnclave].threshold
                    updateRegion.withinRegionHetero = updateRegion.calculateWithinRegionHetero(distance_matrix)
                    del enclaveList[enclave_index]
                    enclave_index = 0

            APartition.shapeindex = APartition.calculateShapeIndex()
            APartition.totalwithinRegionHetero = APartition.calculateWithinRegionHeteroTotal(distance_matrix)
            APartition.id = partition_id
            partitions_list.append(APartition)
            partition_id += 1

    maxp_time2 = datetime.now()
    print(f"    Construction completed in {maxp_time2 - maxp_time1}")
    print(f"    Maximum p found: {max_p}")
    print(f"    Number of partitions: {len(partitions_list)}")

    # 6. Find best partition
    print("\n[6] Finding best partition...")
    minPartitionShape = 1.0
    bestPartition = None

    for partition in partitions_list:
        if partition.p == max_p and partition.shapeindex < minPartitionShape:
            bestPartition = partition
            minPartitionShape = partition.shapeindex

    print(f"    Best shape index: {minPartitionShape:.4f}")
    print(f"    Number of regions: {bestPartition.p}")
    print(f"    Total heterogeneity: {bestPartition.totalwithinRegionHetero:.2f}")

    # 7. Summary
    print("\n[7] Region Summary:")
    print("-" * 60)
    print(f"{'Region':<10} {'Units':<10} {'Threshold Sum':<15} {'Shape Index':<15}")
    print("-" * 60)
    for i, region in enumerate(bestPartition.data):
        print(f"{i+1:<10} {len(region.data):<10} {region.spatialAttrTotal:<15.2f} {region.shapeindex:<15.4f}")
    print("-" * 60)

    # 8. Save visualization
    print("\n[8] Saving visualization...")
    gdf['region'] = bestPartition.label

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    gdf.plot(column=threshold_name, ax=axes[0], legend=True,
             cmap='YlOrRd', edgecolor='black', linewidth=0.5)
    axes[0].set_title(f'Original Data: {threshold_name}')
    axes[0].axis('off')

    gdf.plot(column='region', ax=axes[1], categorical=True,
             cmap='Set3', edgecolor='black', linewidth=0.5, legend=True)
    axes[1].set_title(f'Max-p Regionalization (p={max_p}, Shape Index={minPartitionShape:.4f})')
    axes[1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'result.png'), dpi=150)
    print(f"    Saved to {output_path}/result.png")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    main()
