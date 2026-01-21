#!/bin/bash -l

#SBATCH --partition=normal
#SBATCH --ntasks=2 # asking for 2 cpus
#SBATCH --mem=10G 
#SBATCH --time=1-00:00:00     # 1 day 
#SBATCH --output=jobname_%J_stdout.txt
#SBATCH --error=jobname_%J_stderr.txt
#SBATCH --job-name="maxp_campact" 
#SBATCH --mail-user=selena.feng@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/sfeng/compact_maxp/code/taz

date

hostname


for var in {1..50}
do python C_growth_assign3.py 0 0 0 2 2 100000
done



