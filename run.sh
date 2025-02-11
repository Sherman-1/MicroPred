#!/bin/bash

#PBS -N FineTune
#PBS -q bim
#PBS -l host=node04
#PBS -l ngpus=1
#PBS -l walltime=30000:00:00
#PBS -l mem=32Gb
#PBS -l ncpus=16
#PBS -koed


cd $PBS_O_WORKDIR

gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$gpu_count" -eq 0 ]; then
    echo "No GPU available. Exiting ..."
    exit 1
fi
apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ --bind /scratchlocal/triton_cache /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_MLP/DS_FT.py

