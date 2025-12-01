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
{%- if mail_address is defined %}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={{mail_address}}
{%- endif %}


export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80


export HF_HOME="/hnvme/workspace/b185cb13-lsx/b185cb13-llammlein-1755918661/modular_llammlein/pytorch.sif"
export HF_TOKEN="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"


apptainer exec /hnvme/workspace/b185cb13-lsx/b185cb13-llammlein-1755918661/supergleber_modular.sif python3 src/train.py +model={{model}} +task={{task_name}} {% if grad_accum != 1 %}_grad_accum{{grad_accum}}{% endif %}
