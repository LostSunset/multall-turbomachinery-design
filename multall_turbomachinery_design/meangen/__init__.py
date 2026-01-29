# -*- coding: utf-8 -*-
"""MEANGEN - 一維平均線設計模組。

此模組負責渦輪機械的初始設計，包括：
- 速度三角形計算
- 流道面積設計
- 初始葉片數估算
- 葉片輪廓猜測
"""

from __future__ import annotations

from multall_turbomachinery_design.meangen.data_structures import (
    FlowType,
    GasProperties,
    InputType,
    MachineType,
    MeangenConfig,
    StageDesign,
    VelocityTriangle,
)
from multall_turbomachinery_design.meangen.mean_line_solver import MeanLineSolver

__all__: list[str] = [
    "FlowType",
    "GasProperties",
    "InputType",
    "MachineType",
    "MeangenConfig",
    "MeanLineSolver",
    "StageDesign",
    "VelocityTriangle",
]
