# -*- coding: utf-8 -*-
"""STAGEN - 葉片幾何生成與操作模組。

此模組負責：
- 葉片幾何生成
- 網格細化與堆疊
- 多級葉片組合
- 網格間距控制
- MULTALL 輸入檔案生成
"""

from __future__ import annotations

from .blade_profile import BladeProfileGenerator
from .data_structures import (
    BladeInputType,
    BladeProfile2D,
    BladeRow,
    BladeSection3D,
    GridParameters,
    StackingParameters,
    StagenConfig,
    StreamSurface3D,
    ThicknessParameters,
)
from .grid_generator import GridGenerator
from .io_handler import StagenFileHandler, StagenInputReader, StagenOutputWriter
from .projection import StreamSurfaceProjector
from .solver import StagenSolver, create_simple_blade_row

__all__ = [
    "BladeProfileGenerator",
    "StreamSurfaceProjector",
    "GridGenerator",
    "StagenInputReader",
    "StagenOutputWriter",
    "StagenFileHandler",
    "StagenSolver",
    "create_simple_blade_row",
    "BladeInputType",
    "BladeProfile2D",
    "StreamSurface3D",
    "StackingParameters",
    "BladeSection3D",
    "ThicknessParameters",
    "GridParameters",
    "BladeRow",
    "StagenConfig",
]
