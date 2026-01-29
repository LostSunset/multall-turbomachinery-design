# -*- coding: utf-8 -*-
"""UI - PySide6 圖形使用者介面模組。

此模組提供：
- 主視窗介面
- 各模組的互動式介面
- 參數輸入與驗證
- 結果視覺化
- 檔案管理
"""

from __future__ import annotations

from multall_turbomachinery_design.ui.main_window import MainWindow, main
from multall_turbomachinery_design.ui.panels import (
    MeangenPanel,
    MultallPanel,
    StagenPanel,
)
from multall_turbomachinery_design.ui.widgets import (
    FloatSpinBox,
    IntSpinBox,
    ParameterForm,
    ParameterGroup,
    ResultTable,
    ResultText,
)

__all__: list[str] = [
    "FloatSpinBox",
    "IntSpinBox",
    "MainWindow",
    "MeangenPanel",
    "MultallPanel",
    "ParameterForm",
    "ParameterGroup",
    "ResultTable",
    "ResultText",
    "StagenPanel",
    "main",
]
