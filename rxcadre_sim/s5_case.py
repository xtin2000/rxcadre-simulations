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
TUNED_FUEL_DEPTH_M = 0.37

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

    @classmethod
    def from_rxcadre(cls, chid: str = "s5_base", **overrides: object) -> S5Scenario:
        """Measured S5 values read from the RxCADRE files (see s5_inputs), then any overrides."""
        speed, direction = s5_inputs.wind()
        _, _, half_length = s5_inputs.ignition_line()
        measured = {
            "wind_speed_m_s": speed.value,
            "wind_from_deg": direction.value,
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


def _plot_fuel_vents(
    perimeter: np.ndarray, cell: float, ignition_box: tuple[float, float, float, float]
) -> list[str]:
    """Stair-step the rotated plot polygon into one fuel VENT per same-row run of cells.

    Cells under the ignition line are skipped: FDS rejects a whole VENT that overlaps another.
    """
    polygon = PolygonPath(perimeter)
    x_edges = np.arange(DOMAIN_X_M[0], DOMAIN_X_M[1] + cell / 2, cell)
    y_edges = np.arange(DOMAIN_Y_M[0], DOMAIN_Y_M[1] + cell / 2, cell)
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    vents = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:], strict=True):
        y_centre = (y0 + y1) / 2
        is_inside = polygon.contains_points(np.c_[x_centres, np.full_like(x_centres, y_centre)])
        ign_x0, ign_x1, ign_y0, ign_y1 = ignition_box
        if ign_y0 < y_centre < ign_y1:
            is_inside &= ~((x_centres > ign_x0) & (x_centres < ign_x1))
        run_start = None
        for index, inside in enumerate([*is_inside, False]):
            if inside and run_start is None:
                run_start = index
            elif not inside and run_start is not None:
                vents.append(
                    f"&VENT XB={x_edges[run_start]:.1f},{x_edges[index]:.1f},"
                    f"{y0:.1f},{y1:.1f},0,0, "
                    "SURF_ID='S5 grass' /"
                )
                run_start = None
    return vents


def _ignition_box(scenario: S5Scenario) -> tuple[float, float, float, float]:
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


def build_input(scenario: S5Scenario) -> str:
    """Return the full FDS input text for a scenario."""
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
        f"&WIND DIRECTION={scenario.wind_from_deg}, SPEED={scenario.wind_speed_m_s} /",
        "&DUMP DT_BNDF=10., DT_DEVC=10. /",
        *[
            f"&VENT MB='{side}', SURF_ID='OPEN' /"
            for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMAX")
        ],
        "&SURF ID='BARE', DEFAULT=T, COLOR='GRAY 60' /",
    ]
    body = [
        *_ignition_lines(scenario),
        _fuel_surf_line("S5 grass", scenario, extra="COLOR='GREEN'"),
        *_plot_fuel_vents(load_perimeter_local(), cell, _ignition_box(scenario)),
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
    args = parser.parse_args()

    overrides = {
        "wind_speed_m_s": args.wind_speed,
        "wind_from_deg": args.wind_from,
        "fuel_height_m": args.fuel_height,
        "wind_height_m": args.wind_height,
        "end_time_s": args.end_time,
    }
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
