"""Export an FDS level-set run as a fire-arrival-time grid (NPZ + CSV) and a map image."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import fdsreader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rxcadre_sim.s5_case import (
    DOMAIN_X_M,
    DOMAIN_Y_M,
    IGNITION_CENTER_UTM,
    load_perimeter_local,
    load_sensors_local,
)


def arrival_time_grid(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, arrival[y, x]) with NaN where the fire never arrived."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(run_dir)).slices.filter_by_quantity("LEVEL SET VALUE")[
            0
        ]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    is_burned = frames > 0
    has_burned = is_burned.any(axis=0)
    arrival = np.where(has_burned, times[is_burned.argmax(axis=0)], np.nan).T
    x = np.linspace(*DOMAIN_X_M, arrival.shape[1])
    y = np.linspace(*DOMAIN_Y_M, arrival.shape[0])
    return x, y, arrival


def save_grid(run_dir: Path, x: np.ndarray, y: np.ndarray, arrival: np.ndarray) -> None:
    np.savez_compressed(
        run_dir / "arrival_time.npz",
        x_m=x,
        y_m=y,
        arrival_s=arrival,
        origin_utm16n=np.array(IGNITION_CENTER_UTM),
    )
    xx, yy = np.meshgrid(x, y)
    pd.DataFrame(
        {
            "x_m": xx.ravel(),
            "y_m": yy.ravel(),
            "easting": xx.ravel() + IGNITION_CENTER_UTM[0],
            "northing": yy.ravel() + IGNITION_CENTER_UTM[1],
            "arrival_s": arrival.ravel(),
        }
    ).dropna().to_csv(run_dir / "arrival_time.csv", index=False)


def plot_map(run_dir: Path, x: np.ndarray, y: np.ndarray, arrival: np.ndarray) -> Path:
    figure, axes = plt.subplots(figsize=(7, 7.5))
    minutes = arrival / 60
    mesh = axes.pcolormesh(x, y, minutes, cmap="inferno_r", shading="auto")
    contour_levels = np.arange(2, np.nanmax(minutes), 2)
    contours = axes.contour(x, y, minutes, levels=contour_levels, colors="white", linewidths=0.6)
    axes.clabel(contours, fmt="%d min", fontsize=7)

    perimeter = load_perimeter_local()
    axes.plot(perimeter[:, 0], perimeter[:, 1], color="#2a9d8f", lw=1.5, label="S5 burn block")
    axes.plot([-24.8, 24.8], [0, 0], color="#e63946", lw=3, label="ignition line")

    sensors = load_sensors_local()
    measured = pd.to_timedelta(sensors.arrival_utc).dt.total_seconds()
    axes.scatter(
        sensors.x_local,
        sensors.y_local,
        c="#4cc9f0",
        edgecolor="black",
        s=40,
        zorder=5,
        label="fire sensors (FB)",
    )
    for row, measured_s in zip(sensors.itertuples(), measured - measured.min(), strict=True):
        axes.annotate(
            f"{row.Sensor_ID}\n+{measured_s:.0f}s",
            (row.x_local, row.y_local),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=7,
            color="#4cc9f0",
        )

    axes.set_aspect("equal")
    axes.set_xlabel("east of ignition centre (m)")
    axes.set_ylabel("north of ignition centre (m)")
    axes.set_title(
        f"{run_dir.name}: fire arrival time\n(sensor labels = measured delay after FB18)"
    )
    axes.legend(loc="lower left", fontsize=8)
    figure.colorbar(mesh, ax=axes, label="arrival time after ignition (min)", shrink=0.8)
    figure.tight_layout()
    image_path = run_dir / "arrival_map.png"
    figure.savefig(image_path, dpi=130)
    plt.close(figure)
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        x, y, arrival = arrival_time_grid(run_dir)
        save_grid(run_dir, x, y, arrival)
        burned_m2 = np.isfinite(arrival).sum() * (x[1] - x[0]) * (y[1] - y[0])
        print(
            f"{run_dir.name}: burned ~{burned_m2:,.0f} m2, "
            f"last arrival {np.nanmax(arrival) / 60:.1f} min, "
            f"map {plot_map(run_dir, x, y, arrival)}"
        )


if __name__ == "__main__":
    main()
