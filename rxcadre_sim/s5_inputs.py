"""Read every measured S5 input straight from the RxCADRE files, with where each value came from.

Nothing measured is typed into the code by hand: each value below is read from a downloaded
RxCADRE file (or from data/derived/s5_measurements.json, which prepare_s5.py computes from the raw
anemometer file). Run ``python -m rxcadre_sim.s5_inputs`` to print the table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import shapefile

from rxcadre_sim.paths import DERIVED_DIR, RAW_DIR

MG_HA_TO_KG_M2 = 0.1


@dataclass(frozen=True)
class MeasuredValue:
    """One input value and the exact place it was read from."""

    name: str
    value: float
    unit: str
    dataset: str
    source: str


def fuel_load() -> MeasuredValue:
    path = RAW_DIR / "fuels/2012_S5.csv"
    table = pd.read_csv(path)
    mean_row = table[table.Category == "Mean"].iloc[0]
    mg_per_ha = float(mean_row["Pre_loading_metric_rebinned_TOTAL"])
    return MeasuredValue(
        "fuel amount",
        round(mg_per_ha * MG_HA_TO_KG_M2, 3),
        "kg/m²",
        "RDS-2014-0028",
        f"{path.name}, row 'Mean', column 'Pre_loading_metric_rebinned_TOTAL' = {mg_per_ha} Mg/ha "
        "(average of 25 clip plots), × 0.1 to get kg/m²",
    )


def dead_fuel_moisture() -> MeasuredValue:
    path = RAW_DIR / "fuels/CADRE_2008_2011_2012_Fuel_moistures.csv"
    table = pd.read_csv(path)
    is_s5_litter = (table.Year == 2012) & (table.Site == "S5") & (table.Category == "FM_Litter")
    row = table[is_s5_litter].iloc[0]
    percent = float(row.Mean_FM)
    return MeasuredValue(
        "dead-fuel wetness",
        round(percent / 100, 3),
        "fraction",
        "RDS-2014-0028",
        f"{path.name}, row 2012 / S5 / FM_Litter, column 'Mean_FM' = {percent}% "
        "(oven-dried samples)",
    )


def wind() -> tuple[MeasuredValue, MeasuredValue]:
    path = DERIVED_DIR / "s5_measurements.json"
    burn = json.loads(path.read_text())["wind_during_burn"]
    origin = (
        "computed by prepare_s5.py from Anemometer/Anemometer_Final_v2_only_Splots.csv, "
        f"rows with PlotID S5 during the burn ({burn['samples']} one-second readings, "
        "9 anemometers, 3.05 m above ground)"
    )
    speed = MeasuredValue(
        "wind speed",
        round(burn["mean_speed_m_s"], 1),
        "m/s",
        "RDS-2016-0038",
        f"mean speed {burn['mean_speed_m_s']:.3f} m/s, {origin}",
    )
    direction = MeasuredValue(
        "wind direction (from)",
        float(round(burn["vector_mean_from_deg"])),
        "degrees",
        "RDS-2016-0038",
        f"vector-mean direction {burn['vector_mean_from_deg']:.1f}°, {origin}",
    )
    return speed, direction


def ignition_line() -> tuple[MeasuredValue, MeasuredValue, MeasuredValue]:
    path = RAW_DIR / "survey/SurveyPointInformation.shp"
    reader = shapefile.Reader(str(path), encoding="latin-1")
    points = {}
    for record, shape in zip(reader.records(), reader.shapes(), strict=True):
        sensor = str(record.as_dict()["Sensor_ID"]).strip()
        if sensor in ("S5_IgCenter", "S5_IgEast", "S5_IgWest"):
            points[sensor] = shape.points[0]
    centre, east, west = points["S5_IgCenter"], points["S5_IgEast"], points["S5_IgWest"]
    half_length = round((east[0] - west[0]) / 2, 1)
    where = (
        f"{path.name}, Sensor_ID S5_IgCenter / S5_IgEast / S5_IgWest (survey tags 396 / 395 / 397)"
    )
    return (
        MeasuredValue(
            "ignition line centre, easting",
            round(centre[0], 2),
            "m (UTM 16N)",
            "RDS-2016-0014",
            where,
        ),
        MeasuredValue(
            "ignition line centre, northing",
            round(centre[1], 2),
            "m (UTM 16N)",
            "RDS-2016-0014",
            where,
        ),
        MeasuredValue(
            "ignition line half-length",
            half_length,
            "m",
            "RDS-2016-0014",
            f"(east − west) ÷ 2 = ({east[0]:.2f} − {west[0]:.2f}) ÷ 2",
        ),
    )


def all_measured() -> list[MeasuredValue]:
    return [fuel_load(), dead_fuel_moisture(), *wind(), *ignition_line()]


def main() -> None:
    for item in all_measured():
        print(f"{item.name:32s} {item.value:>12} {item.unit:12s} {item.dataset}  {item.source}")


if __name__ == "__main__":
    main()
