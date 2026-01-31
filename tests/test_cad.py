# -*- coding: utf-8 -*-
"""CAD 模組測試。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestCADAvailability:
    """CAD 可用性測試。"""

    def test_check_cad_available(self) -> None:
        """測試 CAD 可用性檢查。"""
        from multall_turbomachinery_design.cad import check_cad_available

        # 應該返回布林值
        result = check_cad_available()
        assert isinstance(result, bool)

    def test_import_cad_module(self) -> None:
        """測試 CAD 模組導入。"""
        from multall_turbomachinery_design.cad import (
            BladeCADExporter,
            CADExportError,
            check_cad_available,
        )

        assert BladeCADExporter is not None
        assert CADExportError is not None
        assert check_cad_available is not None


# 當 CadQuery 可用時執行的測試
try:
    import cadquery  # noqa: F401

    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


@pytest.mark.skipif(not CADQUERY_AVAILABLE, reason="CadQuery 不可用")
class TestBladeCADExporter:
    """葉片 CAD 導出器測試（需要 CadQuery）。"""

    def test_exporter_initialization(self) -> None:
        """測試導出器初始化。"""
        from multall_turbomachinery_design.cad import BladeCADExporter

        exporter = BladeCADExporter()
        assert exporter is not None
        assert exporter.config is not None

    def test_create_blade_from_sections(self) -> None:
        """測試從截面創建葉片。"""
        from multall_turbomachinery_design.cad.blade_cad import BladeCADExporter, BladeSection

        # 創建簡單的葉片截面
        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points)

        sections = []
        for i, r in enumerate([0.5, 0.6, 0.7]):
            frac = i / 2
            # 簡單的橢圓截面
            x = 0.05 * np.cos(theta)
            y = 0.02 * np.sin(theta)
            z = np.full_like(x, r)

            sections.append(BladeSection(span_fraction=frac, x=x, y=y, z=z))

        exporter = BladeCADExporter()
        blade = exporter.create_blade_from_sections(sections)
        assert blade is not None

    def test_export_step(self, tmp_path: Path) -> None:
        """測試 STEP 導出。"""
        from multall_turbomachinery_design.cad.blade_cad import BladeCADExporter, BladeSection

        # 創建簡單截面
        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points)

        sections = []
        for i, r in enumerate([0.5, 0.7]):
            x = 0.05 * np.cos(theta)
            y = 0.02 * np.sin(theta)
            z = np.full_like(x, r)
            sections.append(BladeSection(span_fraction=i, x=x, y=y, z=z))

        exporter = BladeCADExporter()
        exporter.create_blade_from_sections(sections)

        output_path = tmp_path / "blade.step"
        result = exporter.export(output_path)

        assert result.exists()
        assert result.suffix == ".step"

    def test_export_stl(self, tmp_path: Path) -> None:
        """測試 STL 導出。"""
        from multall_turbomachinery_design.cad.blade_cad import BladeCADExporter, BladeSection

        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points)

        sections = []
        for i, r in enumerate([0.5, 0.7]):
            x = 0.05 * np.cos(theta)
            y = 0.02 * np.sin(theta)
            z = np.full_like(x, r)
            sections.append(BladeSection(span_fraction=i, x=x, y=y, z=z))

        exporter = BladeCADExporter()
        exporter.create_blade_from_sections(sections)

        output_path = tmp_path / "blade.stl"
        result = exporter.export(output_path, format="stl")

        assert result.exists()
        assert result.suffix == ".stl"


@pytest.mark.skipif(not CADQUERY_AVAILABLE, reason="CadQuery 不可用")
class TestConvenienceFunctions:
    """便捷函數測試（需要 CadQuery）。"""

    def test_export_blade_to_step(self, tmp_path: Path) -> None:
        """測試 export_blade_to_step 函數。"""
        from multall_turbomachinery_design.cad.blade_cad import export_blade_to_step

        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points)

        x_coords = [0.05 * np.cos(theta) for _ in range(3)]
        y_coords = [0.02 * np.sin(theta) for _ in range(3)]
        radii = [0.5, 0.6, 0.7]

        output_path = tmp_path / "blade.step"
        result = export_blade_to_step(x_coords, y_coords, radii, output_path)

        assert result.exists()

    def test_export_blade_to_stl(self, tmp_path: Path) -> None:
        """測試 export_blade_to_stl 函數。"""
        from multall_turbomachinery_design.cad.blade_cad import export_blade_to_stl

        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points)

        x_coords = [0.05 * np.cos(theta) for _ in range(3)]
        y_coords = [0.02 * np.sin(theta) for _ in range(3)]
        radii = [0.5, 0.6, 0.7]

        output_path = tmp_path / "blade.stl"
        result = export_blade_to_stl(x_coords, y_coords, radii, output_path)

        assert result.exists()


class TestCADExportError:
    """CAD 導出錯誤測試。"""

    @pytest.mark.skipif(CADQUERY_AVAILABLE, reason="測試 CadQuery 不可用的情況")
    def test_exporter_raises_when_unavailable(self) -> None:
        """測試當 CadQuery 不可用時拋出錯誤。"""
        from multall_turbomachinery_design.cad import BladeCADExporter, CADExportError

        with pytest.raises(CADExportError):
            BladeCADExporter()

    def test_export_without_blade_raises(self) -> None:
        """測試未創建葉片時導出拋出錯誤。"""
        if not CADQUERY_AVAILABLE:
            pytest.skip("CadQuery 不可用")

        from multall_turbomachinery_design.cad import BladeCADExporter, CADExportError

        exporter = BladeCADExporter()
        with pytest.raises(CADExportError):
            exporter.export("test.step")

    def test_create_with_insufficient_sections(self) -> None:
        """測試截面不足時拋出錯誤。"""
        if not CADQUERY_AVAILABLE:
            pytest.skip("CadQuery 不可用")

        from multall_turbomachinery_design.cad import BladeCADExporter, CADExportError
        from multall_turbomachinery_design.cad.blade_cad import BladeSection

        exporter = BladeCADExporter()
        section = BladeSection(
            span_fraction=0.5,
            x=np.array([0, 1]),
            y=np.array([0, 1]),
            z=np.array([0.5, 0.5]),
        )

        with pytest.raises(CADExportError):
            exporter.create_blade_from_sections([section])  # 只有 1 個截面
