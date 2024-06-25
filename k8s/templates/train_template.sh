#!/bin/bash
#SBATCH --job-name="{{job_name}}"
#SBATCH --output=/home/hpc/b185cb/{{user}}/outfiles/out_%j.txt
#SBATCH --error=/home/hpc/b185cb/{{user}}/outfiles/err_%j.txt
#SBATCH --gres=gpu:{{slurm_gpu_count}}
#SBATCH --partition={{slurm_gpu_type}}
{%- if large_gpu %}
#SBATCH --constraint=a100_80
{%- endif %}
#SBATCH --time={{slurm_runtime}}
#SBATCH --chdir={{slurm_path}}
{%- if mail_address is defined %}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={{mail_address}}
{%- endif %}

apptainer exec ~/superkleber.sif python3 src/train.py +model={{model}} +task={{task_name}} +train_args={{slurm_gpu_type}}{% if grad_accum != 1 %}_grad_accum{{grad_accum}}{% endif %}
