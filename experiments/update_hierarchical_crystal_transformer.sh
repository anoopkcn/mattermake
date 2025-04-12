#!/bin/bash
#SBATCH --account=westai0036
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=/p/project1/hai_solaihack/chandran1/mattermake/experiments/update_crystal_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/mattermake/experiments/update_crystal_%j.err

export PROJECT_ROOT="/p/project1/hai_solaihack/chandran1/mattermake"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR}i"

srun --gres=gpu:4 --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    source /p/project1/hai_solaihack/chandran1/mattermake/.venv/bin/activate
    cd /p/home/jusers/chandran1/jureca/hai_solaihack/mattermake/experiments

    python train_hierarchical_crystal_transformer.py hydra.config_name=update_hierarchical_crystal_transformer \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES
"
