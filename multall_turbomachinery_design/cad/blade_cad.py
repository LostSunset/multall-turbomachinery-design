# -*- coding: utf-8 -*-
"""葉片 CAD 輸出模組。

使用 CadQuery 生成葉片的 3D CAD 模型並導出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# 檢查 CadQuery 是否可用
_CADQUERY_AVAILABLE = False
try:
    import cadquery as cq

    _CADQUERY_AVAILABLE = True
except ImportError:
    cq = None  # type: ignore[assignment]


class CADExportError(Exception):
    """CAD 導出錯誤。"""

    pass


def check_cad_available() -> bool:
    """檢查 CAD 功能是否可用。

    Returns:
        True 如果 CadQuery 可用
    """
    return _CADQUERY_AVAILABLE


@dataclass
class BladeSection:
    """葉片截面數據。

    Attributes:
        span_fraction: 跨向位置 (0=根部, 1=葉尖)
        x: X 座標數組 (軸向)
        y: Y 座標數組 (周向/切向)
        z: Z 座標數組 (徑向)
    """

    span_fraction: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]


@dataclass
class BladeCADConfig:
    """葉片 CAD 導出配置。

    Attributes:
        n_sections: 截面數量
        loft_ruled: 是否使用直紋面（否則使用平滑曲面）
        export_format: 導出格式 ('step', 'iges', 'stl')
        stl_tolerance: STL 導出精度
        close_tip: 是否封閉葉尖
        close_root: 是否封閉葉根
        fillet_radius: 圓角半徑（0 表示無圓角）
    """

    n_sections: int = 11
    loft_ruled: bool = False
    export_format: str = "step"
    stl_tolerance: float = 0.01
    close_tip: bool = True
    close_root: bool = True
    fillet_radius: float = 0.0


class BladeCADExporter:
    """葉片 CAD 導出器。

    將葉片幾何數據轉換為 3D CAD 模型並導出。

    Example:
        >>> sections = [BladeSection(...), BladeSection(...)]
        >>> exporter = BladeCADExporter()
        >>> exporter.create_blade(sections)
        >>> exporter.export("blade.step")
    """

    def __init__(self, config: BladeCADConfig | None = None) -> None:
        """初始化導出器。

        Args:
            config: CAD 導出配置

        Raises:
            CADExportError: 如果 CadQuery 不可用
        """
        if not _CADQUERY_AVAILABLE:
            raise CADExportError(
                "CadQuery 不可用。請使用 Python 3.12 並安裝：\n"
                "pip install multall-turbomachinery-design[cad]"
            )

        self.config = config or BladeCADConfig()
        self._blade: Any = None
        self._workplane: Any = None

    def create_blade_from_sections(
        self,
        sections: list[BladeSection],
    ) -> Any:
        """從截面創建葉片 3D 模型。

        Args:
            sections: 葉片截面列表（從根部到葉尖排序）

        Returns:
            CadQuery 實體

        Raises:
            CADExportError: 如果創建失敗
        """
        if len(sections) < 2:
            raise CADExportError("至少需要 2 個截面")

        # 按跨向位置排序
        sections = sorted(sections, key=lambda s: s.span_fraction)

        # 放樣生成實體
        try:
            # 使用 CadQuery 的 sweep/loft 方法
            # 首先創建所有截面的 2D 輪廓
            faces = []
            for section in sections:
                z_height = float(section.z[0])

                # 創建 2D 點序列
                points = []
                for i in range(len(section.x)):
                    pt = (float(section.x[i]), float(section.y[i]))
                    if not points or (
                        abs(pt[0] - points[-1][0]) > 1e-8 or abs(pt[1] - points[-1][1]) > 1e-8
                    ):
                        points.append(pt)

                if len(points) < 3:
                    raise CADExportError(f"截面點數不足: {len(points)}")

                # 確保閉合
                if (
                    abs(points[0][0] - points[-1][0]) > 1e-8
                    or abs(points[0][1] - points[-1][1]) > 1e-8
                ):
                    points.append(points[0])

                # 創建截面面
                wp = cq.Workplane("XY").workplane(offset=z_height)
                face = wp.polyline(points).close().wire()
                faces.append(face)

            # 使用 Solid.makeLoft 進行放樣
            from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

            builder = BRepOffsetAPI_ThruSections(True, not self.config.loft_ruled)

            for face in faces:
                wire = face.val().wrapped
                builder.AddWire(wire)

            builder.Build()
            shape = builder.Shape()

            self._blade = cq.Workplane("XY").newObject([cq.Shape(shape)])

            # 應用圓角
            if self.config.fillet_radius > 0:
                try:
                    self._blade = self._blade.edges().fillet(self.config.fillet_radius)
                except Exception:
                    pass  # 圓角失敗時忽略

        except CADExportError:
            raise
        except Exception as e:
            raise CADExportError(f"放樣失敗: {e}") from e

        return self._blade

    def create_blade_from_profile(
        self,
        x_hub: NDArray[np.float64],
        y_hub: NDArray[np.float64],
        x_tip: NDArray[np.float64],
        y_tip: NDArray[np.float64],
        r_hub: float,
        r_tip: float,
        n_sections: int | None = None,
    ) -> Any:
        """從根部和葉尖輪廓創建葉片。

        使用線性插值生成中間截面。

        Args:
            x_hub: 根部 X 座標
            y_hub: 根部 Y 座標
            x_tip: 葉尖 X 座標
            y_tip: 葉尖 Y 座標
            r_hub: 根部半徑
            r_tip: 葉尖半徑
            n_sections: 截面數量（None 使用配置值）

        Returns:
            CadQuery 實體
        """
        n = n_sections or self.config.n_sections

        sections = []
        for i in range(n):
            frac = i / (n - 1)
            r = r_hub + frac * (r_tip - r_hub)

            # 線性插值截面形狀
            x = x_hub + frac * (x_tip - x_hub)
            y = y_hub + frac * (y_tip - y_hub)
            z = np.full_like(x, r)

            sections.append(
                BladeSection(
                    span_fraction=frac,
                    x=x,
                    y=y,
                    z=z,
                )
            )

        return self.create_blade_from_sections(sections)

    def create_blade_row(
        self,
        sections: list[BladeSection],
        n_blades: int,
        rotate_axis: tuple[float, float, float] = (0, 0, 1),
    ) -> Any:
        """創建整個葉片排（多個葉片）。

        Args:
            sections: 單個葉片的截面列表
            n_blades: 葉片數量
            rotate_axis: 旋轉軸（默認 Z 軸）

        Returns:
            包含所有葉片的 CadQuery 實體
        """
        # 創建單個葉片
        blade = self.create_blade_from_sections(sections)

        # 圓周陣列
        angle = 360.0 / n_blades
        result = blade

        for i in range(1, n_blades):
            rotated = blade.rotate((0, 0, 0), rotate_axis, angle * i)
            result = result.union(rotated)

        self._blade = result
        return result

    def export(
        self,
        filepath: str | Path,
        format: str | None = None,
    ) -> Path:
        """導出 CAD 模型。

        Args:
            filepath: 輸出文件路徑
            format: 導出格式 ('step', 'iges', 'stl')，None 從擴展名推斷

        Returns:
            輸出文件路徑

        Raises:
            CADExportError: 如果尚未創建葉片或導出失敗
        """
        if self._blade is None:
            raise CADExportError("尚未創建葉片模型，請先調用 create_blade_*")

        filepath = Path(filepath)

        # 推斷格式
        if format is None:
            ext = filepath.suffix.lower()
            if ext in (".step", ".stp"):
                format = "step"
            elif ext in (".iges", ".igs"):
                format = "iges"
            elif ext == ".stl":
                format = "stl"
            else:
                format = self.config.export_format

        # 確保目錄存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 導出
        try:
            if format == "step":
                if not filepath.suffix:
                    filepath = filepath.with_suffix(".step")
                cq.exporters.export(self._blade, str(filepath), "STEP")

            elif format == "iges":
                if not filepath.suffix:
                    filepath = filepath.with_suffix(".iges")
                cq.exporters.export(self._blade, str(filepath), "IGES")

            elif format == "stl":
                if not filepath.suffix:
                    filepath = filepath.with_suffix(".stl")
                cq.exporters.export(
                    self._blade,
                    str(filepath),
                    "STL",
                    tolerance=self.config.stl_tolerance,
                )

            else:
                raise CADExportError(f"不支援的格式: {format}")

        except Exception as e:
            raise CADExportError(f"導出失敗: {e}") from e

        return filepath

    def _create_section_wire(self, section: BladeSection) -> Any:
        """創建截面線框。

        Args:
            section: 葉片截面

        Returns:
            CadQuery Wire
        """
        # 創建點列表（排除重複的閉合點）
        points = []
        for i in range(len(section.x)):
            pt = (float(section.x[i]), float(section.y[i]), float(section.z[i]))
            # 避免重複點
            if (
                not points
                or np.sqrt(
                    (pt[0] - points[-1][0]) ** 2
                    + (pt[1] - points[-1][1]) ** 2
                    + (pt[2] - points[-1][2]) ** 2
                )
                > 1e-6
            ):
                points.append(pt)

        # 確保有足夠的點
        if len(points) < 3:
            raise CADExportError(f"截面點數不足: {len(points)}")

        # 使用 polyline 創建閉合線框
        try:
            # 創建在截面 Z 高度的工作平面
            z_height = float(section.z[0])
            wp = cq.Workplane("XY").workplane(offset=z_height)

            # 轉換為 2D 點（在 XY 平面上）
            pts_2d = [(p[0], p[1]) for p in points]

            # 創建閉合的樣條曲線
            wire = wp.spline(pts_2d, periodic=True).val()
            return wire

        except Exception:
            # 備用方案：使用多段線
            try:
                z_height = float(section.z[0])
                wp = cq.Workplane("XY").workplane(offset=z_height)
                pts_2d = [(p[0], p[1]) for p in points]
                wire = wp.polyline(pts_2d).close().val()
                return wire
            except Exception as e:
                raise CADExportError(f"創建截面線框失敗: {e}") from e


def export_blade_to_step(
    x_coords: list[NDArray[np.float64]],
    y_coords: list[NDArray[np.float64]],
    radii: list[float],
    filepath: str | Path,
    n_blades: int = 1,
) -> Path:
    """導出葉片到 STEP 文件的便捷函數。

    Args:
        x_coords: 各截面的 X 座標列表
        y_coords: 各截面的 Y 座標列表
        radii: 各截面的半徑列表
        filepath: 輸出文件路徑
        n_blades: 葉片數量（>1 時創建葉片排）

    Returns:
        輸出文件路徑
    """
    if not check_cad_available():
        raise CADExportError(
            "CadQuery 不可用。請使用 Python 3.12 並安裝：\n"
            "pip install multall-turbomachinery-design[cad]"
        )

    # 創建截面
    sections = []
    for i, (x, y, r) in enumerate(zip(x_coords, y_coords, radii)):
        frac = i / (len(radii) - 1) if len(radii) > 1 else 0.5
        z = np.full_like(x, r)
        sections.append(BladeSection(span_fraction=frac, x=x, y=y, z=z))

    # 創建導出器
    exporter = BladeCADExporter()

    if n_blades > 1:
        exporter.create_blade_row(sections, n_blades)
    else:
        exporter.create_blade_from_sections(sections)

    return exporter.export(filepath, format="step")


def export_blade_to_stl(
    x_coords: list[NDArray[np.float64]],
    y_coords: list[NDArray[np.float64]],
    radii: list[float],
    filepath: str | Path,
    tolerance: float = 0.01,
) -> Path:
    """導出葉片到 STL 文件的便捷函數。

    Args:
        x_coords: 各截面的 X 座標列表
        y_coords: 各截面的 Y 座標列表
        radii: 各截面的半徑列表
        filepath: 輸出文件路徑
        tolerance: STL 精度

    Returns:
        輸出文件路徑
    """
    if not check_cad_available():
        raise CADExportError(
            "CadQuery 不可用。請使用 Python 3.12 並安裝：\n"
            "pip install multall-turbomachinery-design[cad]"
        )

    # 創建截面
    sections = []
    for i, (x, y, r) in enumerate(zip(x_coords, y_coords, radii)):
        frac = i / (len(radii) - 1) if len(radii) > 1 else 0.5
        z = np.full_like(x, r)
        sections.append(BladeSection(span_fraction=frac, x=x, y=y, z=z))

    # 創建導出器
    config = BladeCADConfig(stl_tolerance=tolerance)
    exporter = BladeCADExporter(config)
    exporter.create_blade_from_sections(sections)

    return exporter.export(filepath, format="stl")
