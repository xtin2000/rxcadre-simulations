"""Generate an FDS level-set input file for the RxCADRE 2012 S5 grass burn."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath

from rxcadre_sim import s5_inputs
from rxcadre_sim.paths import DERIVED_DIR, RUNS_DIR

# Reported in RxCADRE field documentation (two igniters walking outward from the line centre).
IGNITER_WALK_SPEED_M_S = 1.34
# The one tuned input: makes the simulated spread match the S5 sensors (calibrate_s5.sh).
TUNED_FUEL_DEPTH_M = 0.42

DOMAIN_X_M = (-80.0, 140.0)
DOMAIN_Y_M = (-180.0, 60.0)
DOMAIN_TOP_M = 6.0


@cache
def ignition_centre_utm() -> tuple[float, float]:
    """Surveyed S5 ignition-line centre (UTM 16N); local x, y are metres from this point."""
    easting, northing, _ = s5_inputs.ignition_line()
    return easting.value, northing.value


@dataclass
class S5Scenario:
    """Inputs for one S5-style level-set run. Use ``from_rxcadre`` to fill the measured values."""

    chid: str
    wind_speed_m_s: float
    wind_from_deg: float
    fuel_load_kg_m2: float
    dead_moisture: float
    ignition_half_length_m: float
    fuel_index: int = 1
    fuel_height_m: float = TUNED_FUEL_DEPTH_M
    wind_height_m: float | None = None
    cell_size_m: float = 2.0
    end_time_s: float = 2400.0
    ignition_xy_m: tuple[float, float] = (0.0, 0.0)
    extra_surf_params: dict[str, float] = field(default_factory=dict)
    # Minute-by-minute measured wind; empty means a steady wind at the mean speed and direction.
    wind_times_s: tuple[float, ...] = ()
    wind_speeds_m_s: tuple[float, ...] = ()
    wind_from_degs: tuple[float, ...] = ()

    @classmethod
    def from_rxcadre(cls, chid: str = "s5_base", **overrides: object) -> S5Scenario:
        """Measured S5 values read from the RxCADRE files (see s5_inputs), then any overrides."""
        speed, direction = s5_inputs.wind()
        _, _, half_length = s5_inputs.ignition_line()
        times, speeds, directions, _ = s5_inputs.wind_series()
        measured = {
            "wind_speed_m_s": speed.value,
            "wind_from_deg": direction.value,
            "wind_times_s": tuple(times),
            "wind_speeds_m_s": tuple(speeds),
            "wind_from_degs": tuple(directions),
            "fuel_load_kg_m2": s5_inputs.fuel_load().value,
            "dead_moisture": s5_inputs.dead_fuel_moisture().value,
            "ignition_half_length_m": half_length.value,
        }
        return cls(chid=chid, **{**measured, **overrides})


def load_perimeter_local() -> np.ndarray:
    """S5 burn-block polygon in metres relative to the ignition-line centre (x east, y north)."""
    points = np.array(json.loads((DERIVED_DIR / "s5_perimeter_xy.json").read_text()))
    return points - np.array(ignition_centre_utm())


def load_sensors_local() -> pd.DataFrame:
    sensors = pd.read_csv(DERIVED_DIR / "s5_fire_sensors_located.csv")
    sensors = sensors[sensors.is_valid].copy()
    centre_x, centre_y = ignition_centre_utm()
    sensors["x_local"] = sensors.x - centre_x
    sensors["y_local"] = sensors.y - centre_y
    return sensors


def _fuel_surf_line(surf_id: str, scenario: S5Scenario, *, extra: str = "") -> str:
    params = [
        f"ID='{surf_id}'",
        f"VEG_LSET_FUEL_INDEX={scenario.fuel_index}",
        f"VEG_LSET_SURF_LOAD={scenario.fuel_load_kg_m2:.4f}",
        f"VEG_LSET_HT={scenario.fuel_height_m:.3f}",
        f"VEG_LSET_M1={scenario.dead_moisture:.3f}",
    ]
    if scenario.wind_height_m is not None:
        params.append(f"VEG_LSET_WIND_HEIGHT={scenario.wind_height_m:.2f}")
    params += [f"{key}={value}" for key, value in scenario.extra_surf_params.items()]
    if extra:
        params.append(extra)
    return "&SURF " + ", ".join(params) + " /"


def plot_cell_mask(
    perimeter: np.ndarray, cell: float, ignition_box: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid edges and a [row, column] mask of cells inside the plot, ignition cells excluded.

    Cells under the ignition line are left out: FDS rejects a whole VENT that overlaps another.
    """
    polygon = PolygonPath(perimeter)
    x_edges = np.arange(DOMAIN_X_M[0], DOMAIN_X_M[1] + cell / 2, cell)
    y_edges = np.arange(DOMAIN_Y_M[0], DOMAIN_Y_M[1] + cell / 2, cell)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centres, y_centres)
    mask = polygon.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ign_x0, ign_x1, ign_y0, ign_y1 = ignition_box
    mask &= ~((xx > ign_x0) & (xx < ign_x1) & (yy > ign_y0) & (yy < ign_y1))
    return x_edges, y_edges, mask


def row_run_vents(
    x_edges: np.ndarray, y_edges: np.ndarray, labels: np.ndarray, surf_ids: list[str]
) -> list[str]:
    """One VENT per same-row run of equal labels; label -1 means no VENT (bare ground)."""
    vents = []
    for row, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:], strict=True)):
        run_start = 0
        row_labels = [*labels[row], -1]
        for index in range(1, len(row_labels)):
            if row_labels[index] == row_labels[run_start]:
                continue
            if row_labels[run_start] >= 0:
                vents.append(
                    f"&VENT XB={x_edges[run_start]:.1f},{x_edges[index]:.1f},"
                    f"{y0:.1f},{y1:.1f},0,0, "
                    f"SURF_ID='{surf_ids[row_labels[run_start]]}' /"
                )
            run_start = index
    return vents


def _plot_fuel_vents(
    perimeter: np.ndarray, cell: float, ignition_box: tuple[float, float, float, float]
) -> list[str]:
    """Stair-step the rotated plot polygon into one fuel VENT per same-row run of cells."""
    x_edges, y_edges, mask = plot_cell_mask(perimeter, cell, ignition_box)
    return row_run_vents(x_edges, y_edges, np.where(mask, 0, -1), ["S5 grass"])


def ignition_box(scenario: S5Scenario) -> tuple[float, float, float, float]:
    cell = scenario.cell_size_m
    centre_x, centre_y = scenario.ignition_xy_m
    half_cells = int(round(scenario.ignition_half_length_m / cell))
    return (
        centre_x - half_cells * cell,
        centre_x + half_cells * cell,
        centre_y,
        centre_y + cell,
    )


def _ignition_lines(scenario: S5Scenario) -> list[str]:
    """Drip-torch line lit from the centre outward by two igniters walking apart."""
    cell = scenario.cell_size_m
    centre_x, centre_y = scenario.ignition_xy_m
    half_cells = int(round(scenario.ignition_half_length_m / cell))
    lines = []
    for offset in range(-half_cells, half_cells):
        x0 = centre_x + offset * cell
        ignite_time = abs(x0 + cell / 2 - centre_x) / IGNITER_WALK_SPEED_M_S
        surf_id = f"IGN {offset:+d}"
        lines.append(
            _fuel_surf_line(
                surf_id, scenario, extra=f"VEG_LSET_IGNITE_TIME={ignite_time:.1f}, COLOR='RED'"
            )
        )
        lines.append(
            f"&VENT XB={x0:.1f},{x0 + cell:.1f},{centre_y:.1f},{centre_y + cell:.1f},0,0, "
            f"SURF_ID='{surf_id}' /"
        )
    return lines


def _sensor_devices(sensors: pd.DataFrame) -> list[str]:
    return [
        f"&DEVC ID='{row.Sensor_ID}', XYZ={row.x_local:.2f},{row.y_local:.2f},0, "
        "QUANTITY='FIRE ARRIVAL TIME', IOR=3 /"
        for row in sensors.itertuples()
    ]


def _wind_lines(scenario: S5Scenario) -> list[str]:
    """Steady wind, or the measured 1-minute series as FDS ramps (speed as a multiple of SPEED)."""
    if not scenario.wind_times_s:
        return [f"&WIND DIRECTION={scenario.wind_from_deg}, SPEED={scenario.wind_speed_m_s} /"]
    lines = [
        f"&WIND DIRECTION={scenario.wind_from_deg}, SPEED={scenario.wind_speed_m_s}, "
        "RAMP_SPEED_T='wind speed', RAMP_DIRECTION_T='wind dir' /"
    ]
    for time, speed in zip(scenario.wind_times_s, scenario.wind_speeds_m_s, strict=True):
        lines.append(
            f"&RAMP ID='wind speed', T={time:g}, F={speed / scenario.wind_speed_m_s:.3f} /"
        )
    for time, direction in zip(scenario.wind_times_s, scenario.wind_from_degs, strict=True):
        lines.append(f"&RAMP ID='wind dir', T={time:g}, F={direction:.1f} /")
    return lines


def build_input(scenario: S5Scenario, *, fuel_lines: list[str] | None = None) -> str:
    """Return the full FDS input text; ``fuel_lines`` replaces the one uniform grass fuel."""
    cell = scenario.cell_size_m
    nx = round((DOMAIN_X_M[1] - DOMAIN_X_M[0]) / cell)
    ny = round((DOMAIN_Y_M[1] - DOMAIN_Y_M[0]) / cell)
    nz = round(DOMAIN_TOP_M / cell)
    header = [
        f"&HEAD CHID='{scenario.chid}', TITLE='RxCADRE 2012 S5 level-set "
        f"(wind {scenario.wind_speed_m_s} m/s from {scenario.wind_from_deg} deg)' /",
        f"&MESH IJK={nx},{ny},{nz}, XB={DOMAIN_X_M[0]},{DOMAIN_X_M[1]},"
        f"{DOMAIN_Y_M[0]},{DOMAIN_Y_M[1]},0,{DOMAIN_TOP_M} /",
        f"&TIME T_END={scenario.end_time_s} /",
        "&MISC LEVEL_SET_MODE=1 /",
        "&RADI RADIATION=F /",
        *_wind_lines(scenario),
        "&DUMP DT_BNDF=10., DT_DEVC=10. /",
        *[
            f"&VENT MB='{side}', SURF_ID='OPEN' /"
            for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX")
        ],
        "&SURF ID='BARE', DEFAULT=T, COLOR='GRAY 60' /",
    ]
    if fuel_lines is None:
        fuel_lines = [
            _fuel_surf_line("S5 grass", scenario, extra="COLOR='GREEN'"),
            *_plot_fuel_vents(load_perimeter_local(), cell, ignition_box(scenario)),
        ]
    body = [
        *_ignition_lines(scenario),
        *fuel_lines,
        "&BNDF QUANTITY='FIRE ARRIVAL TIME' /",
        "&BNDF QUANTITY='LS SPREAD RATE' /",
        "&SLCF AGL_SLICE=0.1, QUANTITY='LEVEL SET VALUE' /",
        *_sensor_devices(load_sensors_local()),
        "&TAIL /",
    ]
    return "\n".join(header + body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--chid", default="s5_base")
    parser.add_argument("--wind-speed", type=float, help="override the measured wind speed")
    parser.add_argument("--wind-from", type=float, help="override the measured wind direction")
    parser.add_argument("--fuel-height", type=float, default=TUNED_FUEL_DEPTH_M)
    parser.add_argument("--wind-height", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=2400.0)
    parser.add_argument(
        "--steady-wind",
        action="store_true",
        help="one constant wind instead of the 1-minute series",
    )
    args = parser.parse_args()

    overrides = {
        "wind_speed_m_s": args.wind_speed,
        "wind_from_deg": args.wind_from,
        "fuel_height_m": args.fuel_height,
        "wind_height_m": args.wind_height,
        "end_time_s": args.end_time,
    }
    if args.steady_wind:
        overrides.update(wind_times_s=(), wind_speeds_m_s=(), wind_from_degs=())
    scenario = S5Scenario.from_rxcadre(
        args.chid, **{key: value for key, value in overrides.items() if value is not None}
    )
    case_dir = args.out_dir / scenario.chid
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / f"{scenario.chid}.fds"
    input_path.write_text(build_input(scenario))
    print(input_path)


if __name__ == "__main__":
    main()
