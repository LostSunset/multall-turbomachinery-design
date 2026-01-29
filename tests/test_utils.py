# -*- coding: utf-8 -*-
"""工具模組測試。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


class TestUnitConverter:
    """單位轉換器測試。"""

    def test_pressure_pa_to_bar(self) -> None:
        """測試 Pa 到 bar。"""
        from multall_turbomachinery_design.utils import PressureUnit, UnitConverter

        result = UnitConverter.convert_pressure(100000, PressureUnit.PA, PressureUnit.BAR)
        assert np.isclose(result, 1.0)

    def test_pressure_bar_to_pa(self) -> None:
        """測試 bar 到 Pa。"""
        from multall_turbomachinery_design.utils import PressureUnit, UnitConverter

        result = UnitConverter.convert_pressure(1.0, PressureUnit.BAR, PressureUnit.PA)
        assert np.isclose(result, 100000)

    def test_pressure_atm_to_pa(self) -> None:
        """測試 atm 到 Pa。"""
        from multall_turbomachinery_design.utils import PressureUnit, UnitConverter

        result = UnitConverter.convert_pressure(1.0, PressureUnit.ATM, PressureUnit.PA)
        assert np.isclose(result, 101325)

    def test_temperature_c_to_k(self) -> None:
        """測試攝氏度到開爾文。"""
        from multall_turbomachinery_design.utils import TemperatureUnit, UnitConverter

        result = UnitConverter.convert_temperature(0, TemperatureUnit.C, TemperatureUnit.K)
        assert np.isclose(result, 273.15)

    def test_temperature_k_to_c(self) -> None:
        """測試開爾文到攝氏度。"""
        from multall_turbomachinery_design.utils import TemperatureUnit, UnitConverter

        result = UnitConverter.convert_temperature(273.15, TemperatureUnit.K, TemperatureUnit.C)
        assert np.isclose(result, 0)

    def test_temperature_f_to_c(self) -> None:
        """測試華氏度到攝氏度。"""
        from multall_turbomachinery_design.utils import TemperatureUnit, UnitConverter

        result = UnitConverter.convert_temperature(32, TemperatureUnit.F, TemperatureUnit.C)
        assert np.isclose(result, 0)

    def test_length_m_to_mm(self) -> None:
        """測試米到毫米。"""
        from multall_turbomachinery_design.utils import LengthUnit, UnitConverter

        result = UnitConverter.convert_length(1.0, LengthUnit.M, LengthUnit.MM)
        assert np.isclose(result, 1000)

    def test_length_in_to_m(self) -> None:
        """測試英寸到米。"""
        from multall_turbomachinery_design.utils import LengthUnit, UnitConverter

        result = UnitConverter.convert_length(1.0, LengthUnit.IN, LengthUnit.M)
        assert np.isclose(result, 0.0254)

    def test_velocity_ms_to_kmh(self) -> None:
        """測試 m/s 到 km/h。"""
        from multall_turbomachinery_design.utils import UnitConverter, VelocityUnit

        result = UnitConverter.convert_velocity(10, VelocityUnit.M_S, VelocityUnit.KM_H)
        assert np.isclose(result, 36)

    def test_mass_flow_kgs_to_kgh(self) -> None:
        """測試 kg/s 到 kg/h。"""
        from multall_turbomachinery_design.utils import MassFlowUnit, UnitConverter

        result = UnitConverter.convert_mass_flow(1, MassFlowUnit.KG_S, MassFlowUnit.KG_H)
        assert np.isclose(result, 3600)

    def test_angular_rpm_to_rads(self) -> None:
        """測試 RPM 到 rad/s。"""
        from multall_turbomachinery_design.utils import AngularVelocityUnit, UnitConverter

        result = UnitConverter.convert_angular_velocity(
            60, AngularVelocityUnit.RPM, AngularVelocityUnit.RAD_S
        )
        assert np.isclose(result, 2 * math.pi)

    def test_angle_deg_to_rad(self) -> None:
        """測試度到弧度。"""
        from multall_turbomachinery_design.utils import AngleUnit, UnitConverter

        result = UnitConverter.convert_angle(180, AngleUnit.DEG, AngleUnit.RAD)
        assert np.isclose(result, math.pi)

    def test_convenience_functions(self) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.utils import (
            convert_length,
            convert_pressure,
            convert_temperature,
        )

        assert np.isclose(convert_pressure(1.0, "bar", "Pa"), 100000)
        assert np.isclose(convert_temperature(0, "°C", "K"), 273.15)
        assert np.isclose(convert_length(1.0, "m", "mm"), 1000)

    def test_helper_methods(self) -> None:
        """測試輔助方法。"""
        from multall_turbomachinery_design.utils import UnitConverter

        assert np.isclose(UnitConverter.rpm_to_rad_s(60), 2 * math.pi)
        assert np.isclose(UnitConverter.rad_s_to_rpm(2 * math.pi), 60)
        assert np.isclose(UnitConverter.bar_to_pa(1), 100000)
        assert np.isclose(UnitConverter.pa_to_bar(100000), 1)
        assert np.isclose(UnitConverter.celsius_to_kelvin(0), 273.15)
        assert np.isclose(UnitConverter.kelvin_to_celsius(273.15), 0)


class TestDataExporter:
    """數據導出器測試。"""

    def test_to_csv_dict(self, tmp_path: Path) -> None:
        """測試導出字典為 CSV。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        path = exporter.to_csv(data, "test.csv")

        assert path.exists()
        content = path.read_text()
        assert "x,y" in content
        assert "1,4" in content

    def test_to_csv_list(self, tmp_path: Path) -> None:
        """測試導出列表為 CSV。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        data = [{"x": 1, "y": 4}, {"x": 2, "y": 5}]
        path = exporter.to_csv(data, "test.csv")

        assert path.exists()
        content = path.read_text()
        assert "x,y" in content

    def test_to_json(self, tmp_path: Path) -> None:
        """測試導出 JSON。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        data = {"name": "test", "values": [1, 2, 3]}
        path = exporter.to_json(data, "test.json")

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["name"] == "test"
        assert loaded["values"] == [1, 2, 3]

    def test_to_json_numpy(self, tmp_path: Path) -> None:
        """測試導出包含 NumPy 數組的 JSON。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        data = {"array": np.array([1.0, 2.0, 3.0])}
        path = exporter.to_json(data, "test.json")

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["array"] == [1.0, 2.0, 3.0]

    def test_to_npz(self, tmp_path: Path) -> None:
        """測試導出 NPZ。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        data = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])}
        path = exporter.to_npz(data, "test.npz")

        assert path.exists()
        loaded = np.load(path)
        assert np.array_equal(loaded["x"], [1, 2, 3])
        assert np.array_equal(loaded["y"], [4, 5, 6])

    def test_to_tecplot(self, tmp_path: Path) -> None:
        """測試導出 Tecplot。"""
        from multall_turbomachinery_design.utils import DataExporter

        exporter = DataExporter(tmp_path)
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        p = np.ones(5) * 101325
        path = exporter.to_tecplot(x, y, {"P": p}, "test.dat")

        assert path.exists()
        content = path.read_text()
        assert "TITLE" in content
        assert "VARIABLES" in content

    def test_convenience_functions(self, tmp_path: Path) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.utils import export_to_csv, export_to_json

        csv_path = export_to_csv({"a": [1, 2]}, tmp_path / "test.csv")
        assert csv_path.exists()

        json_path = export_to_json({"b": 3}, tmp_path / "test.json")
        assert json_path.exists()

    def test_export_performance_report(self, tmp_path: Path) -> None:
        """測試導出性能報告。"""
        from multall_turbomachinery_design.utils import export_performance_report

        perf = {"效率": 0.89, "壓比": 2.5, "功率": 1000.0}
        path = export_performance_report(perf, tmp_path / "report.txt")

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "效率" in content
        assert "0.89" in content


class TestBatchProcessor:
    """批量處理器測試。"""

    def test_parameter_range_from_range(self) -> None:
        """測試從範圍創建參數。"""
        from multall_turbomachinery_design.utils import ParameterRange

        param = ParameterRange.from_range("phi", 0.4, 0.8, 5)
        assert param.name == "phi"
        assert len(param.values) == 5
        assert np.isclose(param.values[0], 0.4)
        assert np.isclose(param.values[-1], 0.8)

    def test_parameter_range_from_list(self) -> None:
        """測試從列表創建參數。"""
        from multall_turbomachinery_design.utils import ParameterRange

        param = ParameterRange.from_list("psi", [1.5, 2.0, 2.5])
        assert param.name == "psi"
        assert len(param.values) == 3

    def test_batch_processor_grid(self) -> None:
        """測試網格掃描。"""
        from multall_turbomachinery_design.utils import BatchProcessor, ParameterRange

        base_config = {"a": 1, "b": 2}

        def compute(config):
            return {"sum": config["a"] + config["b"]}

        processor = BatchProcessor(base_config, compute)
        params = [
            ParameterRange.from_list("a", [1, 2], "a"),
            ParameterRange.from_list("b", [10, 20], "b"),
        ]

        result = processor.run_sweep(params, mode="grid")

        assert result.success_count == 4  # 2 x 2
        assert result.failure_count == 0
        assert len(result.results) == 4

    def test_batch_processor_zip(self) -> None:
        """測試 zip 掃描。"""
        from multall_turbomachinery_design.utils import BatchProcessor, ParameterRange

        base_config = {"a": 1, "b": 2}

        def compute(config):
            return {"sum": config["a"] + config["b"]}

        processor = BatchProcessor(base_config, compute)
        params = [
            ParameterRange.from_list("a", [1, 2, 3], "a"),
            ParameterRange.from_list("b", [10, 20, 30], "b"),
        ]

        result = processor.run_sweep(params, mode="zip")

        assert result.success_count == 3  # 1-1 對應
        assert result.failure_count == 0

    def test_batch_result_to_dataframe_dict(self) -> None:
        """測試結果轉換為 DataFrame 字典。"""
        from multall_turbomachinery_design.utils import BatchResult

        result = BatchResult(
            parameters=[{"a": 1}, {"a": 2}],
            results=[{"x": 10}, {"x": 20}],
            success_count=2,
        )

        df_dict = result.to_dataframe_dict()
        assert df_dict["a"] == [1, 2]
        assert df_dict["x"] == [10, 20]

    def test_parameter_sweep_convenience(self) -> None:
        """測試參數掃描便捷函數。"""
        from multall_turbomachinery_design.utils import ParameterRange, parameter_sweep

        base_config = {"x": 0}

        def compute(config):
            return {"y": config["x"] ** 2}

        params = [ParameterRange.from_list("x", [1, 2, 3], "x")]
        result = parameter_sweep(base_config, params, compute)

        assert result.success_count == 3
        assert result.results[0]["y"] == 1
        assert result.results[1]["y"] == 4
        assert result.results[2]["y"] == 9

    def test_save_batch_results(self, tmp_path: Path) -> None:
        """測試儲存批量結果。"""
        from multall_turbomachinery_design.utils import BatchResult, save_batch_results

        result = BatchResult(
            parameters=[{"a": 1}, {"a": 2}],
            results=[{"x": 10}, {"x": 20}],
            success_count=2,
        )

        paths = save_batch_results(result, tmp_path, prefix="test")

        assert "csv" in paths
        assert "json" in paths
        assert paths["csv"].exists()
        assert paths["json"].exists()


class TestModuleImports:
    """模組導入測試。"""

    def test_import_utils(self) -> None:
        """測試導入 utils 模組。"""
        from multall_turbomachinery_design import utils

        assert utils is not None

    def test_import_unit_converter(self) -> None:
        """測試導入單位轉換器。"""
        from multall_turbomachinery_design.utils import UnitConverter

        assert UnitConverter is not None

    def test_import_data_exporter(self) -> None:
        """測試導入數據導出器。"""
        from multall_turbomachinery_design.utils import DataExporter

        assert DataExporter is not None

    def test_import_batch_processor(self) -> None:
        """測試導入批量處理器。"""
        from multall_turbomachinery_design.utils import BatchProcessor

        assert BatchProcessor is not None
