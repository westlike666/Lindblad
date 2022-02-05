#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-edgrant
#SBATCH --job-name=results/job
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --nodes=4
#SBATCH --mem=128G 

module load python scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install qutip --no-deps 
pip install tqdm

python XY_run_Qutip.py