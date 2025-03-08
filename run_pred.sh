#!/bin/bash

#PBS -N ShuffledData
#PBS -q bim
#PBS -l host=node04
#PBS -l ngpus=1
#PBS -l walltime=72:00:00
#PBS -l mem=450Gb
#PBS -l ncpus=40
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
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/generation/generate_shuffle_data.py