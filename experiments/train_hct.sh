#!/bin/bash
#SBATCH --account=hai_matbind
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=2:00:00
#SBATCH --partition=develbooster
#SBATCH --output=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_hct_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/mattermake/experiments/train_hct_%j.err

export PROJECT_ROOT="/p/project1/hai_solaihack/chandran1/mattermake"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR}i"

srun --gres=gpu:4 --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    source /p/project1/hai_solaihack/chandran1/mattermake/.venv/bin/activate
    cd /p/project1/hai_solaihack/chandran1/mattermake/experiments

    python train_hct.py \
        trainer.max_epochs=50 \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        trainer.devices=4 \
        data.batch_size=32 \
        data.num_workers=4 \
        model.learning_rate=5e-5
"
