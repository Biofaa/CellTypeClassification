#!/bin/bash
#PBS -l select=1:ncpus=76
#PBS -q q02hal
source activate /home/fazzarello/miniforge3/envs/celltype_classification
cd /home/fazzarello/CellTypeClassification
python celltype_classification.py

