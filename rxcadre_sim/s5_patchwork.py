"""Best-guess S5 fuel patchwork: measured fuel shares laid out as random clumps.

RxCADRE measured how much of each fuel S5 had (25 sample plots) but not where it grew, so the
layout is invented: a smoothed random field, thresholded so each fuel covers its measured share.
Run several seeds to see how much the (unknown) layout matters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

from rxcadre_sim.paths import RAW_DIR, RUNS_DIR
from rxcadre_sim.s5_case import (
    TUNED_FUEL_DEPTH_M,
    S5Scenario,
    build_input,
    ignition_box,
    load_perimeter_local,
    load_sensors_local,
    plot_cell_mask,
    row_run_vents,
)

TONS_ACRE_TO_KG_M2 = 0.22417
# Patch size is not measured; clumps a few metres across are a guess for grass, shrubs and oaks.
DEFAULT_PATCH_SIZE_M = 6.0


@dataclass(frozen=True)
class PatchFuel:
    """One fuel class of the patchwork and the FDS settings for it."""

    name: str
    fuel_index: int
    cover_fraction: float
    load_kg_m2: float
    height_m: float | None
    color: str
    live_woody_moisture: float | None = None


def _mean_row(path: str) -> pd.Series:
    table = pd.read_csv(RAW_DIR / path)
    return table[table.Category == "Mean"].iloc[0]


def _measured_moisture(category: str) -> float:
    table = pd.read_csv(RAW_DIR / "fuels/CADRE_2008_2011_2012_Fuel_moistures.csv")
    row = table[(table.Year == 2012) & (table.Site == "S5") & (table.Category == category)]
    return float(row.Mean_FM.iloc[0]) / 100


def patch_fuels(grass_height_m: float, *, separate_litter: bool = False) -> list[PatchFuel]:
    """Shrub/oak clumps plus grass, with cover and load matching the S5 means.

    Litter is spread under every patch. With ``separate_litter`` the ground between plants
    becomes litter-only patches; otherwise it is blended into the grass, because the measured
    bare soil lies between tufts centimetres apart, far below the 2 m grid.
    """
    mean = _mean_row("fuels/2012_S5.csv").apply(pd.to_numeric, errors="coerce")
    load = {
        key: mean[f"Pre_loading_{key}"] * TONS_ACRE_TO_KG_M2
        for key in ("grass", "forb", "shrub", "litter", "woody_1hr_10hr", "oak")
    }
    herb_cover = (mean.Pre_veg_cover_grass + mean.Pre_veg_cover_forb) / 100
    woody_cover = (mean.Pre_veg_cover_shrub + mean.Pre_veg_cover_oak) / 100
    ground_litter = load["litter"] + load["woody_1hr_10hr"]
    woody_height_cm = (
        mean.Pre_veg_cover_shrub * mean.Pre_avg_height_shrub
        + mean.Pre_veg_cover_oak * mean.Pre_avg_height_oak
    ) / (mean.Pre_veg_cover_shrub + mean.Pre_veg_cover_oak)
    if not separate_litter:
        grass_cover = 1 - woody_cover
        return [
            PatchFuel(
                "shrub and oak",
                fuel_index=5,
                cover_fraction=woody_cover,
                load_kg_m2=ground_litter + (load["shrub"] + load["oak"]) / woody_cover,
                height_m=woody_height_cm / 100,
                color="139,90,43",
                live_woody_moisture=_measured_moisture("FM_Shrub"),
            ),
            PatchFuel(
                "grass",
                fuel_index=1,
                cover_fraction=grass_cover,
                load_kg_m2=ground_litter + (load["grass"] + load["forb"]) / grass_cover,
                height_m=grass_height_m,
                color="90,158,69",
            ),
        ]
    return [
        PatchFuel(
            "shrub and oak",
            fuel_index=5,
            cover_fraction=woody_cover,
            load_kg_m2=ground_litter + (load["shrub"] + load["oak"]) / woody_cover,
            height_m=woody_height_cm / 100,
            color="139,90,43",
            live_woody_moisture=_measured_moisture("FM_Shrub"),
        ),
        PatchFuel(
            "grass",
            fuel_index=1,
            cover_fraction=herb_cover,
            load_kg_m2=ground_litter + (load["grass"] + load["forb"]) / herb_cover,
            height_m=grass_height_m,
            color="90,158,69",
        ),
        PatchFuel(
            "litter",
            fuel_index=9,
            cover_fraction=1 - herb_cover - woody_cover,
            load_kg_m2=ground_litter,
            height_m=None,
            color="217,201,138",
        ),
    ]


def patch_labels(
    mask: np.ndarray, fuels: list[PatchFuel], *, seed: int, patch_size_m: float, cell_m: float
) -> np.ndarray:
    """Label each plot cell with a fuel index (-1 outside the plot), clumped by a smooth field."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(mask.shape), sigma=patch_size_m / cell_m / 2)
    ranks = field[mask].argsort()[::-1].argsort() / mask.sum()
    cumulative = np.cumsum([fuel.cover_fraction for fuel in fuels])
    labels = np.full(mask.shape, -1)
    labels[mask] = np.searchsorted(cumulative, ranks, side="right")
    return labels


def _surf_line(fuel: PatchFuel, dead_moisture: float) -> str:
    params = [
        f"ID='{fuel.name}'",
        f"VEG_LSET_FUEL_INDEX={fuel.fuel_index}",
        f"VEG_LSET_SURF_LOAD={fuel.load_kg_m2:.4f}",
        f"VEG_LSET_M1={dead_moisture:.3f}",
    ]
    if fuel.height_m is not None:
        params.append(f"VEG_LSET_HT={fuel.height_m:.3f}")
    if fuel.live_woody_moisture is not None:
        params.append(f"VEG_LSET_MLW={fuel.live_woody_moisture:.3f}")
    params.append(f"RGB={fuel.color}")
    return "&SURF " + ", ".join(params) + " /"


def _plot_layout(path, x_edges, y_edges, labels, fuels) -> None:
    figure, axes = plt.subplots(figsize=(7, 7.5))
    colors = [tuple(int(v) / 255 for v in fuel.color.split(",")) for fuel in fuels]
    shown = np.ma.masked_less(labels, 0)
    axes.pcolormesh(
        x_edges, y_edges, shown, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(fuels) - 0.5
    )
    perimeter = load_perimeter_local()
    axes.plot(perimeter[:, 0], perimeter[:, 1], color="#2a9d8f", lw=1.5)
    axes.plot([-24.8, 24.8], [0, 0], color="#e63946", lw=3, label="ignition line")
    sensors = load_sensors_local()
    axes.scatter(sensors.x_local, sensors.y_local, c="#4cc9f0", edgecolor="black", s=40, zorder=5)
    for fuel, color in zip(fuels, colors, strict=True):
        share = (labels == fuels.index(fuel)).sum() / (labels >= 0).sum()
        axes.scatter([], [], marker="s", s=80, color=color, label=f"{fuel.name} ({share:.0%})")
    axes.set_aspect("equal")
    axes.set_xlabel("east of ignition centre (m)")
    axes.set_ylabel("north of ignition centre (m)")
    axes.set_title(f"{path.parent.name}: best-guess S5 fuel layout")
    axes.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_case(
    chid: str, *, seed: int, patch_size_m: float, grass_height_m: float, separate_litter: bool
) -> None:
    """Write the FDS input and a fuel-layout picture for one patchwork layout."""
    scenario = S5Scenario.from_rxcadre(chid, fuel_height_m=grass_height_m)
    fuels = patch_fuels(grass_height_m, separate_litter=separate_litter)
    x_edges, y_edges, mask = plot_cell_mask(
        load_perimeter_local(), scenario.cell_size_m, ignition_box(scenario)
    )
    labels = patch_labels(
        mask, fuels, seed=seed, patch_size_m=patch_size_m, cell_m=scenario.cell_size_m
    )
    fuel_lines = [
        *[_surf_line(fuel, scenario.dead_moisture) for fuel in fuels],
        *row_run_vents(x_edges, y_edges, labels, [fuel.name for fuel in fuels]),
    ]
    case_dir = RUNS_DIR / chid
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{chid}.fds").write_text(build_input(scenario, fuel_lines=fuel_lines))
    _plot_layout(case_dir / "fuel_layout.png", x_edges, y_edges, labels, fuels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chid", default="s5_patchwork")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--patch-size", type=float, default=DEFAULT_PATCH_SIZE_M)
    parser.add_argument("--grass-height", type=float, default=TUNED_FUEL_DEPTH_M)
    parser.add_argument("--separate-litter", action="store_true")
    args = parser.parse_args()
    for fuel in patch_fuels(args.grass_height, separate_litter=args.separate_litter):
        print(
            f"{fuel.name:14s} FM{fuel.fuel_index:<2d} cover {fuel.cover_fraction:5.1%}  "
            f"load {fuel.load_kg_m2:.3f} kg/m2  height {fuel.height_m}"
        )
    for seed in args.seeds:
        chid = f"{args.chid}_s{seed}"
        write_case(
            chid,
            seed=seed,
            patch_size_m=args.patch_size,
            grass_height_m=args.grass_height,
            separate_litter=args.separate_litter,
        )
        print(RUNS_DIR / chid)


if __name__ == "__main__":
    main()
