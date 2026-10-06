#!/bin/zsh
# Reproduce the S5 calibration: sweep fuel-bed depth with the measured 1-minute wind series
# (the default), plus the earlier steady-wind sweep for comparison, then score every run against
# the measured fire-sensor arrivals.
cd "${0:A:h}/.." || exit 1
python=.venv/bin/python
for depth_cm in 30 32 34 36 38 40 41 42 43 44 46 48 50; do
  $python -m rxcadre_sim.s5_case --chid "calw_h${depth_cm}cm" --fuel-height "0.${depth_cm}" > /dev/null
done
for depth_cm in 30 33 36 37 39 42 45; do
  $python -m rxcadre_sim.s5_case --chid "cal_h${depth_cm}cm_steady" --fuel-height "0.${depth_cm}" \
    --steady-wind > /dev/null
done
scripts/run_fds_batch.sh runs 8
$python -m rxcadre_sim.s5_compare runs/calw_* runs/cal_*_steady
