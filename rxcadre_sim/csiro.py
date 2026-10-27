"""CSIRO grassland fires C064 and F19: NIST's FDS level-set inputs and the measured fire front.

Commands:
  download  fetch NIST's input files (FDS 6.11.1 release) and the measurements into data/raw/csiro/
  setup     copy the inputs into runs/csiro/<case>/ ready for scripts/run_fds_batch.sh
  compare   compare simulated and measured fire-head positions; prints speeds, saves a chart
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rxcadre_sim.paths import RAW_DIR, RUNS_DIR

FIRES = ("C064", "F19")
# NIST runs the level set on three grids: 5 m (LS), 10 m (LS_crude) and 2.5 m (LS_fine) squares.
VARIANTS = ("LS", "LS_crude", "LS_fine")
INPUTS_URL = (
    "https://raw.githubusercontent.com/firemodels/fds/FDS-6.11.1/"
    "Validation/CSIRO_Grassland_Fires/FDS_Input_Files"
)
MEASURED_URL = "https://raw.githubusercontent.com/firemodels/exp/master/CSIRO_Grassland_Fires"
CSIRO_RAW_DIR = RAW_DIR / "csiro"
CSIRO_RUNS_DIR = RUNS_DIR / "csiro"


@dataclass(frozen=True)
class FrontComparison:
    """Measured vs simulated fire-head positions for one run."""

    chid: str
    times_s: np.ndarray
    measured_m: np.ndarray
    simulated_m: np.ndarray

    @property
    def measured_speed_m_s(self) -> float:
        return head_speed(self.times_s, self.measured_m)

    @property
    def simulated_speed_m_s(self) -> float:
        return head_speed(self.times_s, self.simulated_m)


def head_speed(times_s: np.ndarray, positions_m: np.ndarray) -> float:
    """Average speed between the first and last observation."""
    return float((positions_m[-1] - positions_m[0]) / (times_s[-1] - times_s[0]))


def _fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sfL", "-o", str(target), url], check=True)


def download() -> None:
    """Fetch the level-set inputs and the measured front positions for both fires."""
    for fire in FIRES:
        for variant in VARIANTS:
            name = f"Case_{fire}_{variant}.fds"
            _fetch(f"{INPUTS_URL}/{name}", CSIRO_RAW_DIR / "inputs" / name)
        for kind in ("exp", "flame_position_data", "head_depth"):
            name = f"Case_{fire}_{kind}.csv"
            _fetch(f"{MEASURED_URL}/{name}", CSIRO_RAW_DIR / "measured" / name)
    print(f"downloaded to {CSIRO_RAW_DIR}")


def setup() -> None:
    """Put each input in its own run folder, named after its CHID as run_fds_batch.sh expects."""
    for source in sorted((CSIRO_RAW_DIR / "inputs").glob("*.fds")):
        case_dir = CSIRO_RUNS_DIR / source.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, case_dir / source.name)
        print(case_dir)


def compare_run(case_dir: Path) -> FrontComparison:
    """Interpolate the simulated head position (device 'x') at the measured times."""
    fire = case_dir.name.split("_")[1]
    measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire}_exp.csv", skiprows=1)
    devices = pd.read_csv(case_dir / f"{case_dir.name}_devc.csv", skiprows=1)
    simulated = np.interp(measured.Time, devices.Time, devices.x)
    return FrontComparison(
        case_dir.name, measured.Time.to_numpy(), measured.Front.to_numpy(), simulated
    )


def _plot(comparisons: list[FrontComparison], path: Path) -> None:
    figure, axes = plt.subplots(1, len(FIRES), figsize=(11, 4.6))
    for ax, fire in zip(axes, FIRES, strict=True):
        runs = [c for c in comparisons if c.chid.split("_")[1] == fire]
        for comparison in runs:
            devices = pd.read_csv(
                CSIRO_RUNS_DIR / comparison.chid / f"{comparison.chid}_devc.csv", skiprows=1
            )
            ax.plot(devices.Time, devices.x, lw=2, label=f"FDS {comparison.chid.split('_', 2)[2]}")
        if runs:
            ax.scatter(
                runs[0].times_s,
                runs[0].measured_m,
                s=90,
                color="black",
                zorder=5,
                label="real fire",
            )
        ax.set_title(fire)
        ax.set_xlabel("seconds after lighting")
        ax.set_ylabel("fire head distance from lighting line (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def compare() -> None:
    """Print measured vs simulated speeds for every finished run and save a chart."""
    finished = sorted(
        d for d in CSIRO_RUNS_DIR.glob("Case_*") if (d / f"{d.name}_devc.csv").exists()
    )
    comparisons = [compare_run(d) for d in finished]
    for c in comparisons:
        ratio = c.measured_speed_m_s / c.simulated_speed_m_s
        print(
            f"{c.chid:18s} real {c.measured_speed_m_s:.2f} m/s | FDS {c.simulated_speed_m_s:.2f}"
            f" m/s | FDS is {ratio:.1f}x too slow"
            if ratio >= 1
            else f"{c.chid:18s} real {c.measured_speed_m_s:.2f} m/s | FDS "
            f"{c.simulated_speed_m_s:.2f} m/s | FDS is {1 / ratio:.1f}x too fast"
        )
    chart = CSIRO_RUNS_DIR / "front_positions.png"
    _plot(comparisons, chart)
    print(f"chart: {chart}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("download", "setup", "compare"))
    command = parser.parse_args().command
    {"download": download, "setup": setup, "compare": compare}[command]()


if __name__ == "__main__":
    main()
