#!/bin/bash
#SBATCH --job-name=fft_benchmark     # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=8                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:10:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=fft_benchmark%j.out # nom du fichier de sortie
#SBATCH --error=fft_benchmark%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ftb@gpu                   # specify the project
#SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1+nccl-2.8.3-1 nvidia-nsight-systems/2021.1.1

# echo des commandes lancees
set -x

# JZ FIX
export TMPDIR=$JOBSCRATCH
ln -s $JOBSCRATCH /tmp/nvidia

# execution du code avec binding via bind_gpu.sh : 1 GPU pour 1 tache MPI.
#srun nvprof --profile-child-processes -f -o result-%p.nvvp /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u fft_benchmark.py --mesh_shape="b1:2,b2:4" --layout="nx:b1,tny:b1,ny:b2,tnz:b2"
srun --unbuffered --mpi=pmi2 -o fft_%t.log /gpfslocalsup/pub/idrtools/bind_gpu.sh nsys profile --stats=true -t nvtx,cuda,mpi -o result-%q{SLURM_TASK_PID} python -u fft_benchmark.py --mesh_shape="b1:2,b2:4" --layout="nx:b1,tny:b1,ny:b2,tnz:b2"
