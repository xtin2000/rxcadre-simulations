"""CSIRO fires C064 and F19 in FDS's fast level-set mode (mode 1), the mode used for S5.

NIST's validation files run the coupled mode 4. This module rewrites them for mode 1 on the
same 1 m grid as the ELMFIRE runs, for the same cases: NIST's standard grass, the measured grass
height and amount, and the 16-sample field patchwork with the field wind and lighting. FDS's
level set has no input for blade thinness, so the patchwork keeps Eriachne and Themeda alike,
and its wind changes over time but not across the plot (the mean of the two upwind corners).

Commands:
  setup    write runs/csiro_fast/<case>/<case>.fds
           (then: scripts/run_fds_batch.sh runs/csiro_fast 4)
  compare  print fire-head speeds of every engine against the measurements and draw
           runs/csiro_fast/engines_compared.png
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import fdsreader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rxcadre_sim import annaburroo, elmfire_csiro
from rxcadre_sim.csiro import CSIRO_RAW_DIR, CSIRO_RUNS_DIR, head_speed
from rxcadre_sim.elmfire_csiro import (
    FIRES,
    HEAD_STRIP_Y_M,
    Conditions,
    CsiroFire,
    Fuel,
    cases_for,
    upwind_corners,
    weather_times,
)
from rxcadre_sim.paths import RUNS_DIR

RUNS_FAST_DIR = RUNS_DIR / "csiro_fast"
CELL_SIZE_M = 1.0
DOMAIN_TOP_M = 6.0
# Short grass, the closest of FDS's 13 fuel models to the fine CSIRO grass.
MEASURED_GRASS_BASE_MODEL = 1
OUTPUT_LINE = re.compile(r"^&(DEVC|SLCF|BNDF|ISOF)\b")
PLOT_VENT = re.compile(r"&VENT XB= 0\.0,[^/]*SURF_ID='([^']+)' /")
LIGHTING_VENT = re.compile(r"&VENT [^/]*SURF_ID='(IGNITION|LINE)' /\n")
NIST_WIND = re.compile(r"&WIND [^/]*/")
# Field lighting starts as a 2 m spot at the centre of the line, the walkers' starting point.
FIELD_IGNITION_HALF_M = 1.0


def case_name(fire: CsiroFire, fuel: Fuel, conditions: Conditions = Conditions.NIST) -> str:
    suffix = "" if conditions is Conditions.NIST else f"_{conditions.value}"
    return f"Case_{fire.name}_fast_{fuel.value}{suffix}"


def _domain(nist_input: str) -> tuple[float, ...]:
    mesh = re.search(r"&MESH[^/]*XB=([-\d.,\s]+)", nist_input)
    return tuple(float(v) for v in mesh.group(1).split(",")[:4])


def _sample_extents(lines_m: np.ndarray, low: float, high: float) -> dict[float, tuple]:
    """Span of each sample row or column: halfway to its neighbours, the plot edge at the ends."""
    lines = np.unique(lines_m)
    edges = np.concatenate([[low], (lines[:-1] + lines[1:]) / 2, [high]])
    return {line: (edges[i], edges[i + 1]) for i, line in enumerate(lines)}


def patchwork_lines(fire: CsiroFire) -> list[str]:
    """One grass SURF and VENT per field sample: the same nearest-sample patches as ELMFIRE."""
    samples = annaburroo.fuel_samples(fire.name)
    x_spans = _sample_extents(samples.x_m.to_numpy(), 0.0, fire.plot_length_m)
    half = fire.plot_half_width_m
    y_spans = _sample_extents(samples.y_m.to_numpy(), -half, half)
    lines = []
    for n, sample in samples.iterrows():
        (x0, x1), (y0, y1) = x_spans[sample.x_m], y_spans[sample.y_m]
        surf_id = f"GRASS {n + 1:02d}"
        lines += [
            f"&SURF ID='{surf_id}', VEG_LSET_FUEL_INDEX={MEASURED_GRASS_BASE_MODEL}, "
            f"VEG_LSET_SURF_LOAD={sample.load_kg_m2:.4f}, VEG_LSET_HT={sample.height_m}, "
            f"VEG_LSET_M1={fire.dead_moisture_percent / 100:.3f} /",
            f"&VENT XB={x0:g},{x1:g},{y0:g},{y1:g},0.,0., SURF_ID='{surf_id}' /",
        ]
    return lines


def field_lighting_lines(fire: CsiroFire) -> list[str]:
    """A spot at the centre, then the line burns outward at the walking speed."""
    centre, half = FIELD_IGNITION_HALF_M, fire.field_lighting_half_length_m
    return [
        f"&VENT XB=-5.0,0.0,{-centre:g},{centre:g},0.,0., SURF_ID='IGNITION' /",
        f"&VENT XB=-5.0,0.0,{-half:g},{-centre:g},0.,0., SURF_ID='LINE' /",
        f"&VENT XB=-5.0,0.0,{centre:g},{half:g},0.,0., SURF_ID='LINE' /",
    ]


def field_wind_lines(fire: CsiroFire, conditions: Conditions = Conditions.FIELD) -> list[str]:
    """The mean of the two upwind corners every 5 s, as a ramp on NIST's wind profile."""
    measured = annaburroo.upwind_wind(fire.name, corners=upwind_corners(fire, conditions))
    times = weather_times(fire, Conditions.FIELD)
    corner_mean = (measured.right_m_s + measured.left_m_s) / 2
    speeds = np.interp(times, measured.time_s, corner_mean)
    reference = speeds.mean()
    lines = [
        f"&WIND SPEED={reference:.3f}, Z_REF=2., DIRECTION=270., L=-500., Z_0=0.03, "
        "RAMP_SPEED_T='field wind' /"
    ]
    lines += [
        f"&RAMP ID='field wind', T={t:g}, F={u / reference:.3f} /"
        for t, u in zip(times, speeds, strict=True)
    ]
    return lines


def fast_input(fire: CsiroFire, fuel: Fuel, conditions: Conditions = Conditions.NIST) -> str:
    """NIST's level-set input for one fire, switched to mode 1 on a 1 m grid."""
    text = (CSIRO_RAW_DIR / "inputs" / f"Case_{fire.name}_LS.fds").read_text()
    x0, x1, y0, y1 = _domain(text)
    nx, ny = round((x1 - x0) / CELL_SIZE_M), round((y1 - y0) / CELL_SIZE_M)
    nz = round(DOMAIN_TOP_M / CELL_SIZE_M)
    chid = case_name(fire, fuel, conditions)
    text = re.sub(r"CHID='[^']*'", f"CHID='{chid}'", text)
    text = re.sub(
        r"&MESH[^/]*/", f"&MESH IJK={nx},{ny},{nz}, XB={x0},{x1},{y0},{y1},0,{DOMAIN_TOP_M} /", text
    )
    text = text.replace("LEVEL_SET_MODE=4", "LEVEL_SET_MODE=1")
    if fuel is Fuel.MEASURED:
        text = re.sub(
            r"VEG_LSET_FUEL_INDEX\s*=\s*\d+",
            f"VEG_LSET_FUEL_INDEX = {MEASURED_GRASS_BASE_MODEL}, "
            f"VEG_LSET_SURF_LOAD = {fire.grass_load_kg_m2}, VEG_LSET_HT = {fire.grass_height_m}",
            text,
        )
    elif fuel is Fuel.PATCHWORK:
        grass_id = PLOT_VENT.search(text).group(1)
        text = PLOT_VENT.sub("\n".join(patchwork_lines(fire)), text)
        text = re.sub(rf"&SURF ID\s*=\s*'{grass_id}'[^/]*/", "", text)
    if conditions.is_field:
        text = LIGHTING_VENT.sub("", text)
        text = text.replace(
            "&SURF ID='LINE', VEG_LSET_ROS_00 = 1./",
            f"&SURF ID='LINE', VEG_LSET_ROS_00 = {fire.field_walking_speed_m_s:.3f} /\n"
            + "\n".join(field_lighting_lines(fire)),
        )
        text = NIST_WIND.sub("\n".join(field_wind_lines(fire, conditions)), text)
    lines = [line for line in text.splitlines() if not OUTPUT_LINE.match(line)]
    tail = lines.index("&TAIL /")
    lines[tail:tail] = [
        "&RADI RADIATION=F /",
        "&SLCF AGL_SLICE=0.1, QUANTITY='LEVEL SET VALUE' /",
    ]
    return "\n".join(lines) + "\n"


def setup() -> None:
    for fire in FIRES:
        for fuel, conditions in cases_for(fire):
            case_dir = RUNS_FAST_DIR / case_name(fire, fuel, conditions)
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / f"{case_dir.name}.fds").write_text(fast_input(fire, fuel, conditions))
            print(case_dir)


def arrival_times(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, arrival[y, x]) from the level-set slice; NaN where the fire never arrived."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level_set = fdsreader.Simulation(str(case_dir)).slices.filter_by_quantity(
            "LEVEL SET VALUE"
        )[0]
        frames = level_set[0].data
    times = np.asarray(level_set.times)
    is_burned = frames > 0
    arrival = np.where(is_burned.any(axis=0), times[is_burned.argmax(axis=0)], np.nan).T
    x0, x1, y0, y1 = _domain((case_dir / f"{case_dir.name}.fds").read_text())
    return np.linspace(x0, x1, arrival.shape[1]), np.linspace(y0, y1, arrival.shape[0]), arrival


def head_positions(
    x: np.ndarray, y: np.ndarray, arrival: np.ndarray, times_s: np.ndarray
) -> np.ndarray:
    """Furthest burned x inside NIST's head strip (-5 < y < 0), at each time."""
    strip = arrival[(y > HEAD_STRIP_Y_M[0]) & (y < HEAD_STRIP_Y_M[1])]
    positions = []
    for t in times_s:
        burned = np.any(strip <= t, axis=0)
        positions.append(x[burned].max() if burned.any() else np.nan)
    return np.array(positions)


def _engine_curves(fire: CsiroFire) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(label, line style, times, head x) for every finished run of this fire."""
    curves = []
    times = np.arange(0, fire.end_time_s + 1)
    for (fuel, conditions), dash in zip(cases_for(fire), ("-", "--", ":", "-.", "-"), strict=False):
        is_east = conditions is Conditions.FIELD_EAST
        label = elmfire_csiro.case_label(fuel, conditions)
        case_dir = RUNS_FAST_DIR / case_name(fire, fuel, conditions)
        if list(case_dir.glob("*.sf")):
            heads = head_positions(*arrival_times(case_dir), times)
            curves.append(
                (f"FDS fast mode, {label}", ("C3" if is_east else "C1") + dash, times, heads)
            )
        elmfire_dir = elmfire_csiro.RUNS_ELMFIRE_DIR / elmfire_csiro.case_name(
            fire, fuel, conditions
        )
        if list((elmfire_dir / "outputs").glob("time_of_arrival_*.bil")):
            arrival = elmfire_csiro.read_arrival_times(elmfire_dir)
            heads = elmfire_csiro.head_positions(fire, arrival, times)
            curves.append((f"ELMFIRE, {label}", ("C4" if is_east else "C0") + dash, times, heads))
    devc = CSIRO_RUNS_DIR / f"Case_{fire.name}_LS" / f"Case_{fire.name}_LS_devc.csv"
    if devc.exists():
        coupled = pd.read_csv(devc, skiprows=1)
        curves.append(
            (
                "FDS coupled mode (NIST), standard grass",
                "C2:",
                coupled.Time.to_numpy(),
                coupled.x.to_numpy(),
            )
        )
    return curves


def compare() -> None:
    """Print each engine's head speed and draw all head positions over time."""
    figure, axes = plt.subplots(1, len(FIRES), figsize=(12, 5))
    for ax, fire in zip(axes, FIRES, strict=True):
        measured = pd.read_csv(CSIRO_RAW_DIR / "measured" / f"Case_{fire.name}_exp.csv", skiprows=1)
        times = measured.Time.to_numpy()
        print(f"{fire.name}: measured {head_speed(times, measured.Front.to_numpy()):.2f} m/s")
        for label, style, curve_times, heads in _engine_curves(fire):
            speed = head_speed(times, np.interp(times, curve_times, heads))
            print(f"  {label:72s} {speed:.2f} m/s")
            ax.plot(curve_times, heads, style, label=f"{label} ({speed:.2f} m/s)")
        ax.plot(times, measured.Front, "ko", ms=7, label="Real fire")
        ax.set(
            title=f"CSIRO {fire.name}",
            xlabel="Time (s)",
            ylabel="Fire head x (m)",
            ylim=(0, fire.plot_length_m * 1.05),
        )
        ax.legend(fontsize=7.5)
    figure.tight_layout()
    chart = RUNS_FAST_DIR / "engines_compared.png"
    figure.savefig(chart, dpi=150)
    print(f"Chart: {chart}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("setup", "compare"))
    if parser.parse_args().command == "setup":
        setup()
    else:
        compare()


if __name__ == "__main__":
    main()
