# -*- coding: utf-8 -*-
"""視覺化模組。

提供渦輪機械設計結果的視覺化工具：
- 速度三角形圖
- 葉片截面圖
- 流場等值線圖
- 收斂歷史圖
- 性能曲線圖
"""

from __future__ import annotations

from multall_turbomachinery_design.visualization.blade_plots import (
    BladePlotter,
    plot_blade_profile,
    plot_blade_surface,
)
from multall_turbomachinery_design.visualization.convergence_plots import (
    ConvergencePlotter,
    plot_convergence_history,
    plot_residual_history,
)
from multall_turbomachinery_design.visualization.flow_plots import (
    FlowPlotter,
    plot_contour,
    plot_streamlines,
)
from multall_turbomachinery_design.visualization.velocity_triangle import (
    VelocityTrianglePlotter,
    plot_velocity_triangle,
)

__all__: list[str] = [
    "BladePlotter",
    "ConvergencePlotter",
    "FlowPlotter",
    "VelocityTrianglePlotter",
    "plot_blade_profile",
    "plot_blade_surface",
    "plot_contour",
    "plot_convergence_history",
    "plot_residual_history",
    "plot_streamlines",
    "plot_velocity_triangle",
]
