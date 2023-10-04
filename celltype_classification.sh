#!/bin/bash
#PBS -l select=1:ncpus=40
#PBS -q q02daneel
source activate /home/fazzarello/miniforge3/envs/celltype_classification
python celltype_classification.py

