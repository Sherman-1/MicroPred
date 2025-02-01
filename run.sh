#!/bin/bash

#PBS -N PEFT_ProtTrans
#PBS -l host=node04
#PBS -l ngpus=1
#PBS -q bim
#PBS -l walltime=30000:00:00
#PBS -l mem=32Gb
#PBS -l ncpus=8
#PBS -koed


cd $PBS_O_WORKDIR

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ FT.sif python3 peft_PT5.py

