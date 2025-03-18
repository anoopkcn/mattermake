#!/bin/bash
#SBATCH --account=westai0036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=/p/project1/hai_solaihack/chandran1/solai-hackathon-2025/experiments/validate_autoencoder_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/solai-hackathon-2025/experiments/validate_autoencoder_%j.err

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

srun --gres=gpu:1 --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0'
    export PYTHONPATH=''
    source /p/project1/hai_solaihack/chandran1/solai-hackathon-2025/.venv/bin/activate
    cd /p/project1/hai_solaihack/chandran1/solai-hackathon-2025/experiments
    python validate_autoencoder.py
"
