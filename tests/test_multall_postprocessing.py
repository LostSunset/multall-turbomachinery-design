# -*- coding: utf-8 -*-
"""MULTALL 後處理工具測試。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from multall_turbomachinery_design.multall import (
    FlowField,
    GasProperties,
    GasType,
)
from multall_turbomachinery_design.multall.postprocessing import (
    FlowFieldExtractor,
    FlowVisualizationData,
    PerformanceCalculator,
    PerformanceMetrics,
    ResultExporter,
    StagePerformance,
)


class TestPerformanceMetrics:
    """PerformanceMetrics 測試。"""

    def test_default_values(self) -> None:
        """測試默認值。"""
        metrics = PerformanceMetrics()

        assert metrics.mass_flow == 0.0
        assert metrics.power_output == 0.0
        assert metrics.total_to_total_efficiency == 0.0
        assert metrics.pressure_ratio == 0.0

    def test_custom_values(self) -> None:
        """測試自定義值。"""
        metrics = PerformanceMetrics(
            mass_flow=10.0,
            power_output=500.0,
            total_to_total_efficiency=0.90,
            pressure_ratio=4.0,
        )

        assert metrics.mass_flow == 10.0
        assert metrics.power_output == 500.0
        assert metrics.total_to_total_efficiency == 0.90


class TestStagePerformance:
    """StagePerformance 測試。"""

    def test_default_values(self) -> None:
        """測試默認值。"""
        perf = StagePerformance()

        assert perf.stage_number == 1
        assert perf.pressure_ratio == 0.0
        assert perf.isentropic_efficiency == 0.0
        assert perf.reaction == 0.0

    def test_custom_values(self) -> None:
        """測試自定義值。"""
        perf = StagePerformance(
            stage_number=2,
            pressure_ratio=2.5,
            isentropic_efficiency=0.88,
            reaction=0.5,
        )

        assert perf.stage_number == 2
        assert perf.pressure_ratio == 2.5
        assert perf.reaction == 0.5


class TestFlowVisualizationData:
    """FlowVisualizationData 測試。"""

    def test_default_arrays(self) -> None:
        """測試默認數組。"""
        viz = FlowVisualizationData()

        assert viz.x.size == 0
        assert viz.pressure.size == 0
        assert viz.mach.size == 0


class TestPerformanceCalculator:
    """PerformanceCalculator 測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def turbine_flow(self) -> FlowField:
        """創建渦輪流場。"""
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()

        # 入口條件（高壓高溫）
        flow.rho[:, 0:5, :] = 2.5
        flow.vx[:, 0:5, :] = 80.0
        flow.vr[:, 0:5, :] = 0.0
        flow.vt[:, 0:5, :] = 0.0
        flow.p[:, 0:5, :] = 400000.0
        flow.t_static[:, 0:5, :] = 500.0

        # 出口條件（低壓低溫，膨脹後）
        flow.rho[:, 15:, :] = 1.2
        flow.vx[:, 15:, :] = 150.0
        flow.vr[:, 15:, :] = 0.0
        flow.vt[:, 15:, :] = -100.0
        flow.p[:, 15:, :] = 100000.0
        flow.t_static[:, 15:, :] = 350.0

        # 中間區域線性過渡
        for j in range(5, 15):
            frac = (j - 5) / 10.0
            flow.rho[:, j, :] = 2.5 - 1.3 * frac
            flow.vx[:, j, :] = 80.0 + 70.0 * frac
            flow.vt[:, j, :] = -100.0 * frac
            flow.p[:, j, :] = 400000.0 - 300000.0 * frac
            flow.t_static[:, j, :] = 500.0 - 150.0 * frac

        return flow

    def test_init(self, gas: GasProperties) -> None:
        """測試初始化。"""
        calc = PerformanceCalculator(gas)

        assert calc.gas is gas

    def test_compute_overall_performance(self, gas: GasProperties, turbine_flow: FlowField) -> None:
        """測試計算整機性能。"""
        calc = PerformanceCalculator(gas)

        metrics = calc.compute_overall_performance(turbine_flow)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.mass_flow > 0
        assert metrics.pressure_ratio > 1  # 渦輪膨脹比 > 1
        assert metrics.inlet_total_temperature > metrics.exit_total_temperature

    def test_compute_stage_performance(self, gas: GasProperties, turbine_flow: FlowField) -> None:
        """測試計算級性能。"""
        calc = PerformanceCalculator(gas)

        perf = calc.compute_stage_performance(
            turbine_flow, j_inlet=2, j_exit=17, omega=1000.0, r_mean=0.5
        )

        assert isinstance(perf, StagePerformance)
        assert perf.pressure_ratio > 0
        assert perf.flow_coefficient > 0
        assert perf.work_coefficient != 0

    def test_compute_entropy_function(self, gas: GasProperties, turbine_flow: FlowField) -> None:
        """測試計算熵函數。"""
        calc = PerformanceCalculator(gas)

        po_ref = 400000.0
        to_ref = 500.0

        entropy_func = calc.compute_entropy_function(turbine_flow, po_ref, to_ref)

        assert entropy_func.shape == (turbine_flow.im, turbine_flow.jm, turbine_flow.km)
        assert np.all(entropy_func > 0)


class TestFlowFieldExtractor:
    """FlowFieldExtractor 測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def simple_flow(self) -> FlowField:
        """創建簡單流場。"""
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()

        flow.rho[:] = 1.2
        flow.vx[:] = 100.0
        flow.vr[:] = 10.0
        flow.vt[:] = 50.0
        flow.p[:] = 101325.0
        flow.t_static[:] = 300.0

        return flow

    def test_init(self, gas: GasProperties) -> None:
        """測試初始化。"""
        extractor = FlowFieldExtractor(gas)

        assert extractor.gas is gas

    def test_extract_at_j_station(self, gas: GasProperties, simple_flow: FlowField) -> None:
        """測試提取 J 站數據。"""
        extractor = FlowFieldExtractor(gas)

        data = extractor.extract_at_j_station(simple_flow, None, j_index=10)

        assert "rho" in data
        assert "vx" in data
        assert "p" in data
        assert "mach" in data
        assert "t_total" in data
        assert "p_total" in data

        assert data["rho"].shape == (5, 9)

    def test_extract_at_k_surface(self, gas: GasProperties, simple_flow: FlowField) -> None:
        """測試提取 K 流線面數據。"""
        extractor = FlowFieldExtractor(gas)

        data = extractor.extract_at_k_surface(simple_flow, None, k_index=4)

        assert "rho" in data
        assert "velocity" in data
        assert "mach" in data

        assert data["rho"].shape == (5, 20)

    def test_extract_blade_surface_data(self, gas: GasProperties, simple_flow: FlowField) -> None:
        """測試提取葉片表面數據。"""
        extractor = FlowFieldExtractor(gas)

        # 設置壓力面和吸力面壓力差異
        simple_flow.p[0, :, :] = 110000.0  # 壓力面
        simple_flow.p[-1, :, :] = 95000.0  # 吸力面

        data = extractor.extract_blade_surface_data(simple_flow, i_ps=0, i_ss=-1)

        assert "p_ps" in data
        assert "p_ss" in data
        assert "cp_ps" in data
        assert "cp_ss" in data

        # 壓力面壓力係數應該大於吸力面
        assert np.mean(data["cp_ps"]) > np.mean(data["cp_ss"])

    def test_create_visualization_data(self, gas: GasProperties, simple_flow: FlowField) -> None:
        """測試創建可視化數據。"""
        extractor = FlowFieldExtractor(gas)

        viz = extractor.create_visualization_data(simple_flow)

        assert isinstance(viz, FlowVisualizationData)
        assert viz.pressure.shape == (5, 20, 9)
        assert viz.mach.shape == (5, 20, 9)
        assert np.all(viz.mach > 0)


class TestResultExporter:
    """ResultExporter 測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def sample_metrics(self) -> PerformanceMetrics:
        """創建示例性能指標。"""
        return PerformanceMetrics(
            mass_flow=10.0,
            power_output=500.0,
            total_to_total_efficiency=0.90,
            total_to_static_efficiency=0.85,
            pressure_ratio=4.0,
            temperature_ratio=1.3,
            inlet_total_pressure=400000.0,
            inlet_total_temperature=500.0,
            inlet_mach=0.3,
            exit_total_pressure=100000.0,
            exit_static_pressure=90000.0,
            exit_total_temperature=385.0,
            exit_mach=0.5,
            total_pressure_loss_coefficient=0.05,
            entropy_increase=50.0,
        )

    @pytest.fixture
    def simple_flow(self) -> FlowField:
        """創建簡單流場。"""
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()

        flow.rho[:] = 1.2
        flow.vx[:] = 100.0
        flow.vr[:] = 0.0
        flow.vt[:] = 50.0
        flow.p[:] = 101325.0
        flow.t_static[:] = 300.0

        return flow

    def test_init(self, gas: GasProperties) -> None:
        """測試初始化。"""
        exporter = ResultExporter(gas)

        assert exporter.gas is gas
        assert exporter.calculator is not None
        assert exporter.extractor is not None

    def test_export_performance_summary(
        self, gas: GasProperties, sample_metrics: PerformanceMetrics
    ) -> None:
        """測試導出性能摘要。"""
        exporter = ResultExporter(gas)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = f.name

        exporter.export_performance_summary(sample_metrics, output_path)

        # 檢查文件是否創建
        assert Path(output_path).exists()

        # 檢查文件內容
        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        assert "MULTALL 性能計算結果" in content
        assert "質量流量" in content
        assert "功率輸出" in content
        assert "500.00 kW" in content

        # 清理
        Path(output_path).unlink()

    def test_export_station_data_csv(self, gas: GasProperties) -> None:
        """測試導出站點數據為 CSV。"""
        exporter = ResultExporter(gas)

        data = {
            "rho": np.array([[1.2, 1.3], [1.1, 1.2]]),
            "p": np.array([[100000.0, 110000.0], [95000.0, 105000.0]]),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        exporter.export_station_data_csv(data, output_path)

        assert Path(output_path).exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert "i,k,rho,p" in lines[0]
        assert len(lines) == 5  # 表頭 + 4 行數據

        Path(output_path).unlink()

    def test_export_flow_field_binary(self, gas: GasProperties, simple_flow: FlowField) -> None:
        """測試導出流場為二進制格式。"""
        exporter = ResultExporter(gas)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            output_path = f.name

        exporter.export_flow_field_binary(simple_flow, output_path)

        assert Path(output_path).exists()

        # 驗證文件大小合理
        file_size = Path(output_path).stat().st_size
        expected_min_size = 3 * 4 + 5 * 5 * 20 * 9 * 8  # 網格尺寸 + 5 個變量
        assert file_size >= expected_min_size

        Path(output_path).unlink()

    def test_export_convergence_history(self, gas: GasProperties) -> None:
        """測試導出收斂歷史。"""
        exporter = ResultExporter(gas)

        residuals = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        exporter.export_convergence_history(residuals, output_path)

        assert Path(output_path).exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert "iteration,residual" in lines[0]
        assert len(lines) == 6  # 表頭 + 5 行數據

        Path(output_path).unlink()


class TestModuleImports:
    """模組導入測試。"""

    def test_import_performance_calculator(self) -> None:
        """測試導入 PerformanceCalculator。"""
        from multall_turbomachinery_design.multall.postprocessing import (
            PerformanceCalculator,
        )

        assert PerformanceCalculator is not None

    def test_import_flow_field_extractor(self) -> None:
        """測試導入 FlowFieldExtractor。"""
        from multall_turbomachinery_design.multall.postprocessing import (
            FlowFieldExtractor,
        )

        assert FlowFieldExtractor is not None

    def test_import_result_exporter(self) -> None:
        """測試導入 ResultExporter。"""
        from multall_turbomachinery_design.multall.postprocessing import (
            ResultExporter,
        )

        assert ResultExporter is not None
