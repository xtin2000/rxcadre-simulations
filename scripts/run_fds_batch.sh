#!/bin/zsh
# Run every FDS case under a directory (each case is <dir>/<name>/<name>.fds) that
# has not completed yet, N at a time. Usage: run_fds_batch.sh [cases_dir] [parallel_jobs]
# Defaults: scenarios/ and 14 jobs (5 performance + 9 efficiency cores; FDS runs under Rosetta).
source ~/FDS/FDS6/bin/fds_env.zsh
export OMP_NUM_THREADS=1
cases_dir=${1:-"${0:A:h}/../scenarios"}
parallel_jobs=${2:-14}
cd "$cases_dir" || exit 1
for case_dir in */; do
  name=${case_dir%/}
  [[ -f "$name/$name.fds" ]] || continue
  grep -qs "completed successfully" "$name/run.log" || echo "$name"
done | xargs -P "$parallel_jobs" -I{} zsh -c 'cd {} && fds {}.fds > run.log 2>&1; echo "$(date +%T) {} $(tail -1 run.log | grep -q successfully && echo ok || echo FAILED)"'
