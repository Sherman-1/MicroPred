#!/bin/bash

#PBS -N TrainAllModels
#PBS -q bim
#PBS -l host=node04
#PBS -l ngpus=1
#PBS -l walltime=48:00:00
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
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/DB/DB.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/DB_MHAP/DB_MHAP.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_DB/FT_DB.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_DB_MHAP/FT_DB_MHAP.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_MHAP/FT_MHAP.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_MLP/FT_MLP.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/MHAP_MLP/MHAP_MLP.py

apptainer exec --nv --bind /store/EQUIPES/BIM/MEMBERS/simon.herman/ \
    --bind /scratchlocal/triton_cache \
    /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/DeepSpeed.sif \
    python3 /store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/MLP/MLP.py

