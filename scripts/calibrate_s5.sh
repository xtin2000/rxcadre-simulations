#!/bin/zsh
# Reproduce the S5 calibration sweep: fuel-bed depth x wind reference height,
# then score every run against the measured fire-sensor arrivals.
cd "${0:A:h}/.." || exit 1
python=.venv/bin/python
for depth_cm in 15 30 33 36 39 42 45 60; do
  $python -m rxcadre_sim.s5_case --chid "cal_h${depth_cm}cm_whnone" --fuel-height "0.${depth_cm}" > /dev/null
done
for depth_cm in 15 30 45 60; do
  $python -m rxcadre_sim.s5_case --chid "cal_h${depth_cm}cm_wh305" --fuel-height "0.${depth_cm}" \
    --wind-height 3.05 > /dev/null
done
scripts/run_fds_batch.sh runs 8
$python -m rxcadre_sim.s5_compare runs/cal_*
