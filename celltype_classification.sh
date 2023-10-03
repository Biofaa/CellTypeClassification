#!/bin/bash
#PBS -l select=1:ncpus=25
#PBS -q q07daneel
source activate /home/fazzarello/miniforge3/envs/celltype_classification
python celltype_classification.py

