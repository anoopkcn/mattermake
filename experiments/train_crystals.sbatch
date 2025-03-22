#!/bin/bash
#SBATCH --account=westai0036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=6:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_%j.err

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Get master node address
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR}i"

srun --gres=gpu:4 --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    export PYTHONPATH=''
    source /p/project1/hai_solaihack/chandran1/mattermake/.venv/bin/activate
    cd /p/project1/hai_solaihack/chandran1/mattermake/experiments
    python train_crystals.py trainer.max_epochs=500 trainer.num_nodes=$SLURM_JOB_NUM_NODES
"
