#!/bin/bash
#
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=120GB

module load miniconda
source activate py37
python adversarial_ml.py