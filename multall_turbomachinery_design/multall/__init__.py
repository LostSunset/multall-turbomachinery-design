# -*- coding: utf-8 -*-
"""MULTALL - 三維 Navier-Stokes 求解器模組。

此模組負責：
- 3D 流場求解
- 軸流/混流/徑向流支援
- 混合平面模型
- 效率與性能預測
- 逆向設計模式
- 蒸汽渦輪專用查表功能
"""

from __future__ import annotations

from .data_structures import (
    BladeRowGeometry,
    BladeSection,
    BoundaryConditionType,
    ExitBoundary,
    FlowField,
    GasProperties,
    GasType,
    Grid3D,
    GridParameters,
    InletBoundary,
    MixingPlaneParameters,
    MultallConfig,
    SolverParameters,
    TimeStepType,
    ViscousModel,
    ViscousParameters,
)
from .flux import (
    ArtificialViscosity,
    FluxCalculator,
)
from .gas_properties import (
    GasCalculator,
    create_air_calculator,
    create_combustion_gas_calculator,
)
from .inverse_design import (
    BladeDesignSection,
    BladeRedesigner,
    InverseDesignParameters,
    InverseDesignResult,
    InverseDesignSolver,
    InverseDesignType,
)
from .io_handler import (
    MultallFileHandler,
    MultallInputReader,
    MultallOutputWriter,
)
from .mixing_plane import (
    MixingPlaneInterface,
    MixingPlaneModel,
    MixingPlaneType,
    NonReflectingBoundary,
)
from .postprocessing import (
    FlowFieldExtractor,
    FlowVisualizationData,
    PerformanceCalculator,
    PerformanceMetrics,
    ResultExporter,
    StagePerformance,
)
from .solver import (
    MultallSolver,
    create_simple_turbine_solver,
)
from .time_stepping import (
    ConvergenceMonitor,
    TimeStepMethod,
    TimeStepper,
)
from .viscous import (
    MixingLengthModel,
    SpalartAllmarasModel,
    ViscousFluxCalculator,
    WallDistanceCalculator,
)

__all__ = [
    # 數據結構
    "MultallConfig",
    "GasProperties",
    "GasType",
    "GridParameters",
    "SolverParameters",
    "ViscousParameters",
    "ViscousModel",
    "TimeStepType",
    "MixingPlaneParameters",
    "InletBoundary",
    "ExitBoundary",
    "BoundaryConditionType",
    "BladeRowGeometry",
    "BladeSection",
    "FlowField",
    "Grid3D",
    # 氣體性質
    "GasCalculator",
    "create_air_calculator",
    "create_combustion_gas_calculator",
    # I/O
    "MultallInputReader",
    "MultallOutputWriter",
    "MultallFileHandler",
    # 求解器
    "MultallSolver",
    "create_simple_turbine_solver",
    # 通量計算
    "FluxCalculator",
    "ArtificialViscosity",
    # 時間步進
    "TimeStepper",
    "TimeStepMethod",
    "ConvergenceMonitor",
    # 黏性模型
    "MixingLengthModel",
    "SpalartAllmarasModel",
    "ViscousFluxCalculator",
    "WallDistanceCalculator",
    # 混合平面模型
    "MixingPlaneModel",
    "MixingPlaneInterface",
    "MixingPlaneType",
    "NonReflectingBoundary",
    # 逆向設計
    "InverseDesignType",
    "InverseDesignParameters",
    "InverseDesignResult",
    "InverseDesignSolver",
    "BladeRedesigner",
    "BladeDesignSection",
    # 後處理
    "PerformanceMetrics",
    "StagePerformance",
    "FlowVisualizationData",
    "PerformanceCalculator",
    "FlowFieldExtractor",
    "ResultExporter",
]
