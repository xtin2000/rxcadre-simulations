#!/bin/zsh
# Generate, run and export burn scenarios [first, first+count) into burn_dataset/, N at a time.
# Each worker writes its own FDS input, runs FDS, exports a compact .npz and deletes raw output,
# so the dataset grows without the ~80 MB/run of raw slice files. Finished scenarios are skipped,
# so the script can be re-run after an interruption. Keeps the Mac awake while running.
# Usage: run_burn_dataset.sh first count [parallel_jobs]
cd "${0:A:h}/.." || exit 1
source ~/FDS/FDS6/bin/fds_env.zsh
export OMP_NUM_THREADS=1
mkdir -p burn_scenarios burn_dataset
seq "$1" $(($1 + $2 - 1)) | caffeinate -i -s xargs -P "${3:-14}" -n 1 scripts/burn_worker.sh
