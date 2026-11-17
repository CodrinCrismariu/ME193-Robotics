#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
PYMOD=$(cat "/cluster/tufts/c26fa1me0193/ccrism01/mujoco-env/python_module" 2>/dev/null)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "/cluster/tufts/c26fa1me0193/ccrism01/mujoco-env/bin/activate"
export MUJOCO_GL=egl
python train.py Beam-Drive-Straight --env.scene.num-envs 512 --agent.max-iterations 300
