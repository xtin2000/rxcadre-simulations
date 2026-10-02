"""Compare FDS fire arrival times at the S5 fire-behavior sensors with the RxCADRE measurements."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rxcadre_sim.s5_case import load_sensors_local

NO_ARRIVAL = 1.0e5


@dataclass
class Comparison:
    """Model-versus-measurement summary for one run."""

    chid: str
    measured_ros_m_s: float
    model_ros_m_s: float
    rmse_s: float
    implied_ignition_utc: str
    per_sensor: pd.DataFrame


def spread_rate_from_arrivals(x: np.ndarray, y: np.ndarray, arrival_s: np.ndarray) -> float:
    """Head-fire rate of spread from a plane fit t = a + b x + c y (1 / |grad t|)."""
    design = np.c_[np.ones_like(x), x, y]
    coefficients, *_ = np.linalg.lstsq(design, arrival_s, rcond=None)
    return float(1.0 / np.hypot(coefficients[1], coefficients[2]))


def compare_run(run_dir: Path) -> Comparison:
    chid = run_dir.name
    devc = pd.read_csv(run_dir / f"{chid}_devc.csv", skiprows=1)
    final_arrivals = devc.iloc[-1].drop("Time")

    sensors = load_sensors_local().set_index("Sensor_ID")
    sensors["measured_s"] = pd.to_timedelta(sensors.arrival_utc).dt.total_seconds()
    sensors["model_s"] = final_arrivals.reindex(sensors.index).astype(float)
    sensors.loc[sensors.model_s >= NO_ARRIVAL, "model_s"] = np.nan

    reached = sensors.dropna(subset=["model_s"])
    offset = float((reached.measured_s - reached.model_s).mean())
    sensors["model_shifted_s"] = sensors.model_s + offset
    sensors["error_s"] = sensors.model_shifted_s - sensors.measured_s

    return Comparison(
        chid=chid,
        measured_ros_m_s=spread_rate_from_arrivals(
            sensors.x_local.values, sensors.y_local.values, sensors.measured_s.values
        ),
        model_ros_m_s=spread_rate_from_arrivals(
            reached.x_local.values, reached.y_local.values, reached.model_s.values
        ),
        rmse_s=float(np.sqrt(np.nanmean(sensors.error_s**2))),
        implied_ignition_utc=str(pd.to_timedelta(round(offset), unit="s")).split()[-1],
        per_sensor=sensors[["y_local", "measured_s", "model_s", "error_s"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        result = compare_run(run_dir)
        print(
            f"{result.chid:28s} ROS model {result.model_ros_m_s:.3f} vs measured "
            f"{result.measured_ros_m_s:.3f} m/s | timing RMSE {result.rmse_s:5.1f} s | "
            f"implied ignition {result.implied_ignition_utc} UTC"
        )
        if args.detail:
            print(result.per_sensor.round(1).to_string())


if __name__ == "__main__":
    main()
