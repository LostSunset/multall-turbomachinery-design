# -*- coding: utf-8 -*-
"""UI 面板模組。

提供各模組的面板：
- MeangenPanel: 平均線設計面板
- StagenPanel: 葉片幾何面板
- MultallPanel: 求解器面板
"""

from __future__ import annotations

from multall_turbomachinery_design.ui.panels.meangen_panel import MeangenPanel
from multall_turbomachinery_design.ui.panels.multall_panel import MultallPanel
from multall_turbomachinery_design.ui.panels.stagen_panel import StagenPanel

__all__: list[str] = [
    "MeangenPanel",
    "MultallPanel",
    "StagenPanel",
]
