# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Contract-first G2 grasp learning pipeline.

The modules in this package are intentionally independent from the legacy
v1--v44 launch hierarchy.  Legacy components may be adapted into this package,
but this package never imports a launcher as an authority.
"""

from .data_contract import (
    ACTOR_ALLOWED_INPUTS,
    CoordinateFrame,
    DatasetSource,
    EpisodeRole,
    G2ArtifactLineage,
    G2DataContract,
    G2GoalConditioningContract,
    G2_RUNTIME_JOINT_ORDER,
    GateOutcome,
    GateRecord,
    SchemaVersion,
    TensorSource,
    TensorSpec,
    Unit,
)

__all__ = [
    "ACTOR_ALLOWED_INPUTS",
    "CoordinateFrame",
    "DatasetSource",
    "EpisodeRole",
    "G2ArtifactLineage",
    "G2DataContract",
    "G2GoalConditioningContract",
    "G2_RUNTIME_JOINT_ORDER",
    "GateOutcome",
    "GateRecord",
    "SchemaVersion",
    "TensorSource",
    "TensorSpec",
    "Unit",
]
