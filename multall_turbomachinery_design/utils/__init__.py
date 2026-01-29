# -*- coding: utf-8 -*-
"""工具函數模組。

提供各種實用工具：
- 單位轉換
- 數據導出
- 批量處理
"""

from __future__ import annotations

from multall_turbomachinery_design.utils.batch import (
    BatchProcessor,
    BatchResult,
    ParameterRange,
    parameter_sweep,
    save_batch_results,
)
from multall_turbomachinery_design.utils.export import (
    DataExporter,
    export_performance_report,
    export_to_csv,
    export_to_json,
)
from multall_turbomachinery_design.utils.units import (
    AngleUnit,
    AngularVelocityUnit,
    LengthUnit,
    MassFlowUnit,
    PressureUnit,
    TemperatureUnit,
    UnitConverter,
    VelocityUnit,
    convert_length,
    convert_pressure,
    convert_temperature,
)

__all__: list[str] = [
    # 單位轉換
    "AngleUnit",
    "AngularVelocityUnit",
    "LengthUnit",
    "MassFlowUnit",
    "PressureUnit",
    "TemperatureUnit",
    "UnitConverter",
    "VelocityUnit",
    "convert_length",
    "convert_pressure",
    "convert_temperature",
    # 數據導出
    "DataExporter",
    "export_performance_report",
    "export_to_csv",
    "export_to_json",
    # 批量處理
    "BatchProcessor",
    "BatchResult",
    "ParameterRange",
    "parameter_sweep",
    "save_batch_results",
]
