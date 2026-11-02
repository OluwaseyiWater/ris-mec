#!/bin/bash
# =============================================================================
#  Full profile on a Digital Research Alliance of Canada cluster.
#
#  CPU-ONLY BY DESIGN. Measured on this workload, only ~40% of wall time is
#  inside torch update steps (the strategist's inner loop is 97% NumPy), so even
#  an infinitely fast GPU caps the end-to-end speed-up at 1.66x -- and the policy
#  is a 64x64 MLP, small enough that CUDA kernel-launch overhead often makes it
#  slower than CPU. Requesting GPUs would cost allocation and queue time for
#  nothing. What the job needs is CPU cores: the (K, method, seed) grid is 840+
#  independent runs.
#
#  Submit from a directory on ~/scratch or ~/projects (NOT /home -- quota):
#      sbatch scripts/alliance_full.sh
# =============================================================================
#SBATCH --job-name=ris-mec-full
#SBATCH --account=def-CHANGEME          # your allocation: `sacctmgr show assoc user=$USER format=account`
#SBATCH --nodes=1                       # single node: workers share one filesystem + model cache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # = --jobs below
#SBATCH --mem-per-cpu=2G                # ~1 GB peak RSS per worker, 2G is safe headroom
#SBATCH --time=03:00:00                 # ~50 min measured at 32 cores; margin for slower nodes
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGEME@example.com

set -euo pipefail
echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK}  started=$(date -Is)"

module purge
module load StdEnv/2023 python/3.11        # `module spider python` to see what is available

# Build the venv on the node's local SSD. --no-index uses the Alliance's own
# wheelhouse, so this works even where compute nodes have no internet.
virtualenv --no-download "${SLURM_TMPDIR}/venv"
source "${SLURM_TMPDIR}/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy pandas matplotlib pyyaml torch gymnasium
pip install --no-index stable_baselines3 || pip install stable-baselines3   # may not be in the wheelhouse

# One torch thread per worker: 32 workers x 4 threads would oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

python -c "import numpy, torch, stable_baselines3 as sb3, gymnasium; \
print('numpy', numpy.__version__, 'torch', torch.__version__, \
'sb3', sb3.__version__, 'gym', gymnasium.__version__)"

# --cache writes trained policies to results/model_cache/, so if this job hits the
# time limit, resubmitting resumes instead of retraining from scratch.
srun python -m experiments.run_all \
    --profile full \
    --jobs "${SLURM_CPUS_PER_TASK}" \
    --threads 1 \
    --auto-v \
    --cache

echo "finished=$(date -Is)"
