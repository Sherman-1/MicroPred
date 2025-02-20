#!/bin/bash

#PBS -N GenDataPred
#PBS -q bim
#PBS -l host=node04
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -l mem=128Gb
#PBS -l ncpus=8
#PBS -koed


cd $PBS_O_WORKDIR

gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$gpu_count" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
    if [ "$gpu_count" -eq 0 ]; then
        echo "No GPU available. Exiting ..."
        exit 1
    fi
fi
apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/generation/generate_pred_data.py

