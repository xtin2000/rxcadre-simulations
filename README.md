# RxCADRE Simulations

Physics-based wildfire spread scenarios for drone-based spread prediction research.
NIST's [Fire Dynamics Simulator (FDS)](https://github.com/firemodels/fds) level-set
model is calibrated against the **RxCADRE 2012 S5** prescribed grass burn (Eglin
AFB, Florida, 1 Nov 2012) and then used to generate fictitious, but physically
grounded, fire scenarios. The scenarios serve as ground truth for the sibling
project `drone_lab/wildfire_spread_prediction` (its `--fds-scenario` mode).

![S5 calibrated fire arrival map](docs/s5_base_arrival_map.png)

## Results

**S5 calibration.** All inputs are measured except the fuel-bed depth, the least
constrained Rothermel input, which was swept (`scripts/calibrate_s5.sh`).

| Input | Value | Source |
|---|---|---|
| Plot | official S5 perimeter, 19,957 m² | RDS-2014-0031 |
| Fuel load | 0.282 kg/m² (dead fuel moisture 8.6 %) | RDS-2014-0028 |
| Wind | 2.8 m/s from 335° (9 anemometers at 3 m, during the burn) | RDS-2016-0038 |
| Ignition | surveyed 50 m line, lit from the centre outward at 1.34 m/s | RDS-2016-0014 |
| Fuel model | Rothermel-Albini #1 (short grass), depth calibrated to **0.37 m** | this work |

| | Measured | FDS, uncalibrated (0.30 m) | FDS, calibrated (0.37 m) |
|---|---|---|---|
| Head-fire rate of spread | 0.123 m/s | 0.091 m/s | **0.120 m/s** |
| Arrival-time RMSE at 6 sensors | — | 50 s | **34 s** |
| Implied ignition time (UTC) | 18:05–18:07 (extrapolated) | 18:02 | **18:05** |

Using the 3 m anemometer wind directly as mid-flame wind (`VEG_LSET_WIND_HEIGHT=3.05`)
over-predicts spread 2.5–10×, so the FDS default wind adjustment is used. Most of
the remaining error comes from two sensors 10 m apart that the real fire reached in
the reverse order: fuel or gust heterogeneity that a uniform fuel bed cannot reproduce.

**100 fictitious scenarios** (`scenarios/`) on a 400 × 400 m grid of 2 m cells
(matching the drone simulation's `TerrainGrid`), sampled from the measured
RxCADRE small-plot (S3–S9) ranges:

| Parameter | Range |
|---|---|
| Wind | 1.5–4.5 m/s from 335° ± 45°, one gradual shift of up to ±40°, gusts up to ±25 % |
| Dead fuel moisture | 6.4–10 % |
| Fuel load | 0.24–0.41 kg/m² |
| Fuel depth | 0.30–0.44 m (calibrated depth scaled by each plot's grass height) |
| Fuel patchiness | spatially correlated, 0–35 % strength, 8–40 m patch size |
| Ignition | point or walked drip-torch line, upwind of centre |
| Firebreak | 6 m bare road across the wind in ~30 % of scenarios |

## Installation

Requires macOS or Linux, Python 3.12+, and FDS 6.11 (installed from the
[FDS releases](https://github.com/firemodels/fds/releases); on Apple Silicon the
x86 build runs under Rosetta). The scripts expect FDS's environment at
`~/FDS/FDS6/bin/fds_env.zsh`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

## Running the pipeline

```bash
# 1. Download the public RxCADRE datasets (~400 MB, S5 slices only)
.venv/bin/python -m rxcadre_sim.download

# 2. Derive S5 perimeter, sensor arrivals, wind and measured spread rate
.venv/bin/python -m rxcadre_sim.prepare_s5

# 3. Build, run and score the calibrated S5 base case (~15 s)
.venv/bin/python -m rxcadre_sim.s5_case --chid s5_base
scripts/run_fds_batch.sh runs
.venv/bin/python -m rxcadre_sim.s5_compare runs/s5_base --detail
.venv/bin/python -m rxcadre_sim.s5_map runs/s5_base

# Optional: reproduce the calibration sweep
scripts/calibrate_s5.sh

# 4. Generate and run the fictitious scenarios (~30 min for 100 on 14 cores)
.venv/bin/python -m rxcadre_sim.scenarios --count 100
scripts/run_fds_batch.sh scenarios 14

# 5. Export ground-truth grids for the drone simulation
.venv/bin/python -m rxcadre_sim.scenario_export scenarios/scn* \
    --out-dir ../drone_lab/wildfire_spread_prediction/data/fds_scenarios
```

Each exported `scnNNN.npz` holds fire arrival time, spread velocity, fuel load,
fuel depth and moisture per cell, plus the wind time series and the scenario spec.

## Testing

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

## Project layout

```
rxcadre_sim/   download, S5 data preparation, S5 case/compare/map, scenario generation/export
scripts/       FDS batch runner and calibration sweep
data/derived/  small S5 inputs and validation targets (tracked); data/raw/ is downloaded
runs/          S5 base and calibration cases (inputs tracked, FDS outputs ignored)
scenarios/     100 scenario inputs + specs (FDS outputs ignored)
docs/          figures
```

## Limitations

- **Level-set mode 1:** the fire front follows Rothermel-Albini spread with uniform,
  prescribed wind. No fire-induced winds, no spotting, flat terrain.
- **Calibrated on one burn:** only head-fire arrival was measured, so flank and
  backing spread (fire shape) are not validated. The RxCADRE airborne infrared
  data (RDS-2017-0027) could check this.
- **Moisture capped at 10 %:** fuel model 1 has a 12 % dead-fuel moisture of
  extinction. Plot S7 (13 % litter moisture) burned in reality, but would not in
  this model.
- **Fires reach the edge:** most scenarios reach the 400 m domain edge within
  20–40 min, which limits long forecast horizons.

## Data sources

All RxCADRE data are public, from the USDA Forest Service Research Data Archive:

- Ottmar, R.D.; Restaino, J.C. 2014. *RxCADRE 2008, 2011, and 2012: Ground fuel measurements from prescribed fires.* https://doi.org/10.2737/RDS-2014-0028
- Jimenez, D.M.; Butler, B.W. 2016. *RxCADRE 2012: In-situ fire behavior measurements.* https://doi.org/10.2737/RDS-2016-0038
- *RxCADRE 2012: Instrument and infrared target survey locations and attributes.* https://doi.org/10.2737/RDS-2016-0014
- *RxCADRE 2008, 2011, and 2012: Burn blocks.* https://doi.org/10.2737/RDS-2014-0031
- Background: *Modeling Low Intensity Fires: Lessons Learned from 2012 RxCADRE.* Atmosphere 12(2):139, 2021. https://doi.org/10.3390/atmos12020139
