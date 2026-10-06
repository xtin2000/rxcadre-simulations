from __future__ import annotations

import pytest

from rxcadre_sim import s5_inputs
from rxcadre_sim.paths import RAW_DIR

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "fuels/2012_S5.csv").exists(), reason="raw RxCADRE data not downloaded"
)


def test_measured_s5_values_are_read_from_the_rxcadre_files():
    assert s5_inputs.fuel_load().value == pytest.approx(0.282)
    assert s5_inputs.dead_fuel_moisture().value == pytest.approx(0.086)


def test_every_measured_value_names_its_dataset_and_source():
    for item in s5_inputs.all_measured():
        assert item.dataset.startswith("RDS-")
        assert item.source
