# -*- coding: utf-8 -*-
"""MULTALL 渦輪機械設計系統 - Python/PySide6 實現。

本專案基於原始 FORTRAN 77 MULTALL 系統重新實現。
原始系統來源: https://sites.google.com/view/multall-turbomachinery-design/

主要模組:
- meangen: 一維平均線設計
- stagen: 葉片幾何生成與操作
- multall: 三維 Navier-Stokes 求解器
- ui: PySide6 圖形使用者介面
- utils: 工具函數與輔助類別
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "LostSunset"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
