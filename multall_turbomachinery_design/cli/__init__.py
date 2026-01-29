# -*- coding: utf-8 -*-
"""命令行介面模組。

提供 MULTALL 渦輪機械設計系統的命令行工具：
- multall: 主命令
- multall meangen: 平均線設計
- multall stagen: 葉片幾何生成
- multall solve: 運行求解器
- multall plot: 視覺化結果
"""

from __future__ import annotations

from multall_turbomachinery_design.cli.main import app, main

__all__: list[str] = [
    "app",
    "main",
]
