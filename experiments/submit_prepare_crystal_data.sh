#!/bin/bash
#SBATCH --account=westai0036
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --partition=dc-hwai
#SBATCH --output=/p/project1/hai_solaihack/chandran1/mattermake/experiments/prepare_data_%j.out
#SBATCH --error=/p/project1/hai_solaihack/chandran1/mattermake/experiments/prepare_data_%j.err

export TOKENIZERS_PARALLELISM=false

srun --gres=gpu:0 --nodes=1 --ntasks-per-node=1 --cpu-bind=none bash -c "
    source /p/project1/hai_solaihack/chandran1/mattermake/.venv/bin/activate
    cd /p/project1/hai_solaihack/chandran1/mattermake

    # Run the preparation script with direct arguments
    python -m mattermake.scripts.prepare_crystal_data \
        --input_csv="/p/project1/hai_solaihack/datasets/alex_mp_20/alex_mp_20/train.csv" \
        --output_dir="/p/project1/hai_solaihack/chandran1/mattermake/data/structure_tokens" \
        --standardize
"
