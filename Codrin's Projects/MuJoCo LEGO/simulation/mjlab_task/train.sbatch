#!/bin/bash
#SBATCH --job-name=beam-drive-straight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=train-%j.out

# Submit from simulation/mjlab_task on a GPU cluster with ~/mujoco-env installed.
cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
nvidia-smi -L
python train.py Beam-Drive-Straight --env.scene.num-envs 512 --agent.max-iterations 300
