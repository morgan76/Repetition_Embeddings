#!/bin/bash
#SBATCH --job-name=test_job 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --partition A100
#SBATCH --cpus-per-task=10 
#SBATCH --distribution=block:block  
#SBATCH --time=20:00:00
#SBATCH --output=train_%j.log


set -x

source ~/anaconda3/bin/activate

conda init bash
conda activate segmentation_4
cd /tsi/data_doctorants/mbuisson/TASLP_24_Rep/src/

srun python3 train.py --data_path ../../msaf_/datasets/music_collection --val_data_path ../../msaf_/datasets/Harmonix  --output_dir results/exp_4 --temperature_positives .1 --temperature_negatives .1 --temperature_loss .1 --max_len 1500 --n_training_samples 200000
