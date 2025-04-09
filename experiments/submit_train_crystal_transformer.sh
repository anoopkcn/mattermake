#!/bin/bash
#SBATCH --account=westai0036
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_crystal_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_crystal_%j.err

# Set environment variables
export PROJECT_ROOT="/p/project1/hai_solaihack/chandran1/mattermake"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Get master node address for multi-node training
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR}i"

# Run the training
srun --gres=gpu:4 --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    source /p/project1/hai_solaihack/chandran1/mattermake/.venv/bin/activate
    cd /p/project1/hai_solaihack/chandran1/mattermake

    # Run training with Hydra configuration
    python experiments/train_crystal_transformer.py \
        trainer.max_epochs=100 \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        trainer.devices=4 \
        data.batch_size=32 \
        data.num_workers=4 \
        continuous_predictions=true \
        +continuous_regression_weight=0.5 \
        logger.wandb.name="continuous_crystal_transformer"
"
