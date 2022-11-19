#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=Muframex
#SBATCH --output=results.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=giovanni.lopez@cimat.mx
nvidia-smi
cd $(pwd)
source /opt/anaconda3_titan/bin/activate
conda activate torch
hostname
python src/test.py
date

