# Max-p Regionalization with Compactness Constraints

A Python implementation of the max-p regionalization algorithm with compactness constraints for spatial analysis and geographic regionalization tasks.

## Overview

This project implements a spatially constrained clustering algorithm that:
- Maximizes the number of regions (max-p) while meeting a minimum threshold constraint
- Optimizes region compactness using moment of inertia-based shape indices
- Supports multiple construction strategies and local search optimization via simulated annealing

## Features

- **Region Growth**: Greedy construction of regions from seed polygons with optional compactness-aware unit selection
- **Enclave Assignment**: Assignment of unassigned areas to neighboring regions with compactness optimization
- **Local Search**: Simulated annealing-based optimization to improve region compactness
- **Flexible Parameters**: Configurable flags for enabling/disabling compactness considerations at each stage

## Requirements

- Python 3.6+
- numpy
- scipy
- pandas
- geopandas
- libpysal
- pysal
- matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/xf37/maxp_compact.git
cd maxp_compact

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

For a step-by-step tutorial, see the **[demo notebook](examples/demo.ipynb)** which uses sample data from libpysal.

## Usage

### Basic Usage

```bash
python src/maxp_compact.py <flag_rg> <flag_ea> <flag_ls> <randomGrow> <randomAssign> <threshold>
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `flag_rg` | Enable compactness in region growth (0 or 1) |
| `flag_ea` | Enable compactness in enclave assignment (0 or 1) |
| `flag_ls` | Enable compactness in local search (0 or 1) |
| `randomGrow` | Number of top candidates for random selection during growth |
| `randomAssign` | Number of top candidates for random selection during enclave assignment |
| `threshold` | Minimum population threshold for each region |

### Example

```bash
# Run with all compactness optimizations enabled
python src/maxp_compact.py 1 1 1 2 2 100000

# Run without compactness optimization (baseline)
python src/maxp_compact.py 0 0 0 2 2 100000
```

### Using SLURM (HPC Clusters)

An example SLURM batch script is provided in `examples/run_slurm.sh`. Modify the paths and parameters as needed for your cluster environment.

## Input Data

The algorithm expects a shapefile with:
- Polygon geometries
- A threshold attribute (e.g., population) for the minimum constraint
- An attribute for measuring within-region heterogeneity (e.g., median household income)

## Output

- CSV file with regionalization results including:
  - Maximum p value (number of regions)
  - Shape index (compactness measure)
  - Within-region heterogeneity
  - Computation time statistics

## Algorithm Details

### Shape Index (Compactness Measure)

The compactness is measured using a normalized moment of inertia-based shape index:

```
Shape Index = 1 - (Area^2) / (2 * pi * Moment of Inertia)
```

A lower shape index indicates a more compact (circular) region.

### Three-Phase Approach

1. **Construction Phase (Region Growth)**: Grows regions from random seeds by iteratively adding neighboring units
2. **Enclave Assignment**: Assigns remaining unassigned areas to neighboring regions
3. **Local Search**: Refines the solution using simulated annealing

## Citation

If you use this code in your research, please cite:

```bibtex
@article{feng2022maxp,
  title={The max-p-compact-regions problem},
  author={Feng, Xin and Rey, Sergio and Wei, Ran},
  journal={Transactions in GIS},
  volume={26},
  number={2},
  pages={717--734},
  year={2022},
  publisher={Wiley}
}
```

Feng, X., Rey, S., & Wei, R. (2022). The max‐p‐compact‐regions problem. *Transactions in GIS*, 26(2), 717-734.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Xin Feng
- Sergio Rey
- Ran Wei

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
