from __future__ import annotations

import re

import numpy as np
import pytest

from rxcadre_sim.csiro_fds_full import (
    FULL_INPUTS_DIR,
    MPI_PROCESSES,
    FullCase,
    Grid,
    fds_direction,
    full_input,
    furthest_head,
)
from rxcadre_sim.elmfire_csiro import FIRES

C064, F19 = FIRES
needs_nist_inputs = pytest.mark.skipif(
    not (FULL_INPUTS_DIR / "Case_C064.fds").exists(),
    reason="run `python -m rxcadre_sim.csiro_fds_full download` first",
)


@needs_nist_inputs
def test_every_mesh_names_one_of_the_mpi_processes():
    processes = re.findall(r"MPI_PROCESS=(\d+)", full_input(FullCase(C064)))
    assert len(processes) == 36 and {int(p) for p in processes} == set(range(MPI_PROCESSES))


@needs_nist_inputs
def test_patchwork_gives_each_field_sample_its_own_grass():
    assert full_input(FullCase(C064)).count("&INIT PART_ID='blade of grass") == 16


@needs_nist_inputs
def test_mixed_grass_keeps_both_blade_thicknesses():
    text = full_input(FullCase(F19))
    assert "SURFACE_VOLUME_RATIO = 9770." in text and "SURFACE_VOLUME_RATIO = 12240." in text


@needs_nist_inputs
def test_lighting_spreads_at_the_walkers_pace():
    assert "SPREAD_RATE=1.571" in full_input(FullCase(F19))


@needs_nist_inputs
def test_wind_follows_the_field_record():
    assert "RAMP_SPEED_T='field wind'" in full_input(FullCase(C064))


@needs_nist_inputs
def test_one_metre_grid_keeps_nists_coarse_mesh():
    assert "&MESH IJK=40,40,20," in full_input(FullCase(F19, Grid.ONE_METRE))


def test_head_stays_at_the_furthest_point_reached_on_the_plot():
    hottest_x = np.array([10.0, 60.0, 109.5, 45.5])
    assert list(furthest_head(hottest_x, 100.0)) == [10.0, 60.0, 100.0, 100.0]


def test_station_direction_turns_compass_bearings_into_simulation_axes():
    from_east, from_south, from_north = fds_direction([90.0, 180.0, 0.0])
    assert (from_east, from_south, from_north) == (270.0, 0.0, 180.0)


@needs_nist_inputs
def test_turning_wind_case_adds_the_direction_ramp():
    text = full_input(FullCase(F19, Grid.ONE_METRE, turns_with_station=True))
    assert "RAMP_DIRECTION_T='station direction'" in text
