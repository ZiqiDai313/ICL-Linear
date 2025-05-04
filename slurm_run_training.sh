#!/bin/bash
#SBATCH --array=131-133
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=ace
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 1-24:00 # time requested (D-HH:MM)


pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate sparseml
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd $(pwd)/TransformerICL

SLURM_ARRAY_TASK_ID=$1
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ./config/hyper.txt)
seed=$(echo $cfg | cut -f 1 -d ' ')
lr=$(echo $cfg | cut -f 2 -d ' ')
model=$(echo $cfg | cut -f 3 -d ' ')
clip_t=$(echo $cfg | cut -f 4 -d ' ')
clip_r=$(echo $cfg | cut -f 5 -d ' ')
bs=$(echo $cfg | cut -f 6 -d ' ')
max_iter=$(echo $cfg | cut -f 7 -d ' ')
hidden_dim_factor=$(echo $cfg | cut -f 8 -d ' ')
scheduler=$(echo $cfg | cut -f 9 -d ' ')
var=$(echo $cfg | cut -f 10 -d ' ')
alpha=$(echo $cfg | cut -f 11 -d ' ')
fix_min=$(echo $cfg | cut -f 12 -d ' ')
alg=$(echo $cfg | cut -f 13 -d ' ')

python main.py \
        --log_dir ./checkpoints/nla/solver/ \
        --only_eval False \
        --model ${model} \
        --lr ${lr} \
        --alg adam \
        --dim 9 \
        --batch_size ${bs} \
        --max_iters ${max_iter} \
        --condition_number 5 \
        --n_layer 3 \
        --clip ${clip_r} \
        --n_head 1 \
        --alpha ${alpha} \
        --fix_min ${fix_min} \
        --seed ${seed} \
        --clip_type ${clip_t} \
        --hidden_dim_factor ${hidden_dim_factor} \
        --scheduler ${scheduler} \
        --var ${var} \
        --alg ${alg} \
        --eval_hessian True
