# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import pytest

from geniesim.rl.sac.privileged_geometry_oracle import (
    PRIMARY_PAD_BODIES,
    _measure_primary_pad_cube_geometry,
)


def test_exact_mesh_vertices_define_table_clearance() -> None:
    inner = np.asarray(
        [[[0.0, -0.05, 0.735], [0.01, -0.05, 0.736], [0.0, -0.04, 0.737]]],
        dtype=np.float64,
    )
    outer = np.asarray(
        [[[0.0, 0.05, 0.732], [0.01, 0.05, 0.733], [0.0, 0.04, 0.734]]],
        dtype=np.float64,
    )
    result = _measure_primary_pad_cube_geometry(
        meshes_world_m={
            PRIMARY_PAD_BODIES[0]: inner,
            PRIMARY_PAD_BODIES[1]: outer,
        },
        authority={},
        cube_center_world_m=(0.1, 0.0, 0.76),
        cube_quat_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        cube_half_extents_m=(0.01, 0.01, 0.01),
        table_surface_height_m=0.73,
    )
    assert result["inner_link4_table_clearance_mm"] == pytest.approx(5.0)
    assert result["outer_link4_table_clearance_mm"] == pytest.approx(2.0)
    assert result["minimum_primary_pad_table_clearance_mm"] == pytest.approx(2.0)
    assert result["student_observation_field_count"] == 0
