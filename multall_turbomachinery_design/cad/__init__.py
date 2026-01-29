# -*- coding: utf-8 -*-
"""CAD 輸出模組。

提供葉片幾何的 CAD 格式導出功能。
支援 STEP、IGES、STL 等格式。

注意：此模組需要 CadQuery 依賴，目前僅支援 Python 3.12。
安裝方式：pip install multall-turbomachinery-design[cad]
"""

from __future__ import annotations

from multall_turbomachinery_design.cad.blade_cad import (
    BladeCADExporter,
    CADExportError,
    check_cad_available,
)

__all__: list[str] = [
    "BladeCADExporter",
    "CADExportError",
    "check_cad_available",
]
