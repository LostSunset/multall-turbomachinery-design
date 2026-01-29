# -*- coding: utf-8 -*-
"""UI 元件模組。

提供可重用的 UI 元件：
- 參數輸入元件
- 驗證元件
- 結果顯示元件
"""

from __future__ import annotations

from multall_turbomachinery_design.ui.widgets.parameter_input import (
    FloatSpinBox,
    IntSpinBox,
    ParameterForm,
    ParameterGroup,
)
from multall_turbomachinery_design.ui.widgets.result_display import (
    ResultTable,
    ResultText,
)

__all__: list[str] = [
    "FloatSpinBox",
    "IntSpinBox",
    "ParameterForm",
    "ParameterGroup",
    "ResultTable",
    "ResultText",
]
