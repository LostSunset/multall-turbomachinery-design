# -*- coding: utf-8 -*-
"""MULTALL 逆向設計模組測試。"""

from __future__ import annotations

import numpy as np
import pytest

from multall_turbomachinery_design.multall import (
    FlowField,
    GasProperties,
    GasType,
)
from multall_turbomachinery_design.multall.inverse_design import (
    BladeDesignSection,
    BladeRedesigner,
    InverseDesignParameters,
    InverseDesignResult,
    InverseDesignSolver,
    InverseDesignType,
)


class TestInverseDesignType:
    """InverseDesignType 枚舉測試。"""

    def test_enum_values(self) -> None:
        """測試枚舉值。"""
        assert InverseDesignType.PRESSURE_LOADING == 1
        assert InverseDesignType.EXIT_ANGLE == 2
        assert InverseDesignType.BLADE_FORCE == 3


class TestInverseDesignParameters:
    """InverseDesignParameters 測試。"""

    def test_default_values(self) -> None:
        """測試默認值。"""
        params = InverseDesignParameters()

        assert params.enabled is False
        assert params.design_type == InverseDesignType.PRESSURE_LOADING
        assert params.max_iterations == 100
        assert params.convergence_tolerance == 1e-4
        assert params.angle_relaxation == 0.1
        assert params.rotation_relaxation == 0.5

    def test_custom_values(self) -> None:
        """測試自定義值。"""
        params = InverseDesignParameters(
            enabled=True,
            design_type=InverseDesignType.EXIT_ANGLE,
            max_iterations=50,
            target_exit_angle=np.radians(-60.0),
        )

        assert params.enabled is True
        assert params.design_type == InverseDesignType.EXIT_ANGLE
        assert params.max_iterations == 50
        assert np.isclose(params.target_exit_angle, np.radians(-60.0))


class TestBladeDesignSection:
    """BladeDesignSection 測試。"""

    def test_creation(self) -> None:
        """測試創建。"""
        section = BladeDesignSection(k_index=5)

        assert section.k_index == 5
        assert section.x_stream.size == 0
        assert section.r_stream.size == 0
        assert section.rt_upper.size == 0

    def test_with_data(self) -> None:
        """測試帶數據創建。"""
        section = BladeDesignSection(
            k_index=3,
            x_stream=np.array([0.0, 0.1, 0.2]),
            r_stream=np.array([0.5, 0.5, 0.5]),
        )

        assert section.k_index == 3
        assert len(section.x_stream) == 3
        assert len(section.r_stream) == 3


class TestInverseDesignResult:
    """InverseDesignResult 測試。"""

    def test_default_values(self) -> None:
        """測試默認值。"""
        result = InverseDesignResult()

        assert result.converged is False
        assert result.iterations == 0
        assert result.current_exit_angle == 0.0
        assert result.rotation_angle == 0.0

    def test_custom_values(self) -> None:
        """測試自定義值。"""
        result = InverseDesignResult(
            converged=True,
            iterations=10,
            current_exit_angle=-1.0,
            compatible_exit_angle=-1.05,
            rotation_angle=0.05,
        )

        assert result.converged is True
        assert result.iterations == 10
        assert result.current_exit_angle == -1.0
        assert result.rotation_angle == 0.05


class TestBladeRedesigner:
    """BladeRedesigner 測試。"""

    @pytest.fixture
    def redesigner(self) -> BladeRedesigner:
        """創建重新設計器。"""
        return BladeRedesigner(smooth_factor=0.25, smooth_iterations=3)

    def test_init(self, redesigner: BladeRedesigner) -> None:
        """測試初始化。"""
        assert redesigner.smooth_factor == 0.25
        assert redesigner.smooth_iterations == 3

    def test_smooth_data(self, redesigner: BladeRedesigner) -> None:
        """測試數據平滑。"""
        x = np.linspace(0, 1, 10)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        y_smooth = redesigner._smooth_data(x, y)

        # 平滑後變化應該更小
        assert np.std(y_smooth) < np.std(y)

    def test_create_section(self, redesigner: BladeRedesigner) -> None:
        """測試創建葉片截面。"""
        # 流線面座標（直線）
        n_pts = 20
        x_stream = np.linspace(0, 0.1, n_pts)
        r_stream = np.full(n_pts, 0.5)
        relative_spacing = np.ones(n_pts)

        # 葉片參數
        n_le = 4
        n_te = 15
        n_new = 10
        frac_new = np.linspace(0, 1, n_new)
        beta_new = np.linspace(-30, -60, n_new)  # 度
        thick_upper = np.array([0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.015, 0.01, 0.005])
        thick_lower = thick_upper * 0.8

        section = redesigner.create_section(
            k_index=5,
            x_stream=x_stream,
            r_stream=r_stream,
            relative_spacing=relative_spacing,
            n_le=n_le,
            n_te=n_te,
            frac_new=frac_new,
            beta_new=beta_new,
            thick_upper_frac=thick_upper,
            thick_lower_frac=thick_lower,
        )

        assert section.k_index == 5
        assert len(section.x_stream) > 0
        assert len(section.rt_upper) > 0
        assert len(section.rt_thickness) > 0


class TestInverseDesignSolver:
    """InverseDesignSolver 測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def params(self) -> InverseDesignParameters:
        """創建逆向設計參數。"""
        return InverseDesignParameters(
            enabled=True,
            design_type=InverseDesignType.EXIT_ANGLE,
            target_exit_angle=np.radians(-60.0),
            j_leading_edge=5,
            j_trailing_edge=15,
        )

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()

        # 初始化流場
        flow.rho[:] = 1.2
        flow.vx[:] = 100.0
        flow.vr[:] = 0.0
        flow.vt[:] = -50.0  # 負切向速度（渦輪）
        flow.p[:] = 101325.0
        flow.t_static[:] = 300.0

        return flow

    def test_init(self, gas: GasProperties, params: InverseDesignParameters) -> None:
        """測試初始化。"""
        solver = InverseDesignSolver(gas, params)

        assert solver.gas is gas
        assert solver.params is params
        assert len(solver.history) == 0

    def test_compute_current_conditions(
        self, gas: GasProperties, params: InverseDesignParameters, small_flow: FlowField
    ) -> None:
        """測試計算當前條件。"""
        solver = InverseDesignSolver(gas, params)

        conditions = solver.compute_current_conditions(
            small_flow,
            j_le=params.j_leading_edge,
            j_te=params.j_trailing_edge,
            omega=1000.0,
            n_blades=50,
        )

        assert "vx_in" in conditions
        assert "vx_out" in conditions
        assert "vt_out" in conditions
        assert "exit_angle" in conditions
        assert "blade_force" in conditions

    def test_compute_compatible_angle(
        self, gas: GasProperties, params: InverseDesignParameters
    ) -> None:
        """測試計算相容角度。"""
        solver = InverseDesignSolver(gas, params)

        conditions = {
            "vx_out": 100.0,
            "vt_out": -50.0,
            "vt_in": 0.0,
            "vm_out": 100.0,
            "rho_out": 1.2,
            "exit_angle": -0.5,
        }

        angle = solver.compute_compatible_angle(
            conditions, target_force=1000.0, omega=1000.0, r_avg=0.5
        )

        # 角度應該在合理範圍內
        assert -np.pi / 2 < angle < np.pi / 2

    def test_compute_rotation_angle(
        self, gas: GasProperties, params: InverseDesignParameters
    ) -> None:
        """測試計算旋轉角度。"""
        solver = InverseDesignSolver(gas, params)

        current = np.radians(-55.0)
        target = np.radians(-60.0)

        rotation = solver.compute_rotation_angle(current, target)

        # 旋轉角度應該是差值乘以鬆弛因子
        expected = (target - current) * params.rotation_relaxation
        assert np.isclose(rotation, expected)

    def test_iterate(
        self, gas: GasProperties, params: InverseDesignParameters, small_flow: FlowField
    ) -> None:
        """測試迭代。"""
        solver = InverseDesignSolver(gas, params)

        result = solver.iterate(small_flow, omega=1000.0, n_blades=50)

        assert isinstance(result, InverseDesignResult)
        assert result.iterations == 1
        assert len(solver.history) == 1

    def test_iterate_pressure_loading(self, gas: GasProperties, small_flow: FlowField) -> None:
        """測試基於壓力載荷的迭代。"""
        params = InverseDesignParameters(
            enabled=True,
            design_type=InverseDesignType.PRESSURE_LOADING,
            j_leading_edge=5,
            j_trailing_edge=15,
        )
        solver = InverseDesignSolver(gas, params)

        # 目標壓力分佈
        n_pts = 11
        target_ps = np.full(n_pts, 110000.0)
        target_ss = np.full(n_pts, 100000.0)

        result = solver.iterate(
            small_flow,
            target_pressure_ps=target_ps,
            target_pressure_ss=target_ss,
            omega=1000.0,
            n_blades=50,
        )

        assert isinstance(result, InverseDesignResult)

    def test_iterate_blade_force(self, gas: GasProperties, small_flow: FlowField) -> None:
        """測試基於葉片力的迭代。"""
        params = InverseDesignParameters(
            enabled=True,
            design_type=InverseDesignType.BLADE_FORCE,
            target_blade_force=500.0,
            j_leading_edge=5,
            j_trailing_edge=15,
        )
        solver = InverseDesignSolver(gas, params)

        result = solver.iterate(small_flow, omega=1000.0, n_blades=50)

        assert isinstance(result, InverseDesignResult)

    def test_apply_geometry_modification(
        self, gas: GasProperties, params: InverseDesignParameters
    ) -> None:
        """測試應用幾何修正。"""
        solver = InverseDesignSolver(gas, params)

        rt_upper = np.array([0.0, 0.01, 0.02, 0.03, 0.02, 0.01, 0.0])
        rt_thickness = np.array([0.0, 0.005, 0.01, 0.015, 0.01, 0.005, 0.0])
        r_surface = np.full(7, 0.5)
        rotation_angle = np.radians(2.0)

        rt_upper_new, rt_thickness_new = solver.apply_geometry_modification(
            rt_upper, rt_thickness, rotation_angle, r_surface
        )

        # 上表面應該偏移
        assert not np.allclose(rt_upper_new, rt_upper)
        # 厚度應該不變
        assert np.allclose(rt_thickness_new, rt_thickness)

    def test_reset_history(
        self, gas: GasProperties, params: InverseDesignParameters, small_flow: FlowField
    ) -> None:
        """測試重置歷史。"""
        solver = InverseDesignSolver(gas, params)

        # 執行幾次迭代
        for _ in range(3):
            solver.iterate(small_flow, omega=1000.0, n_blades=50)

        assert len(solver.history) == 3

        solver.reset_history()
        assert len(solver.history) == 0

    def test_get_convergence_history(
        self, gas: GasProperties, params: InverseDesignParameters, small_flow: FlowField
    ) -> None:
        """測試獲取收斂歷史。"""
        solver = InverseDesignSolver(gas, params)

        # 執行幾次迭代
        for _ in range(5):
            solver.iterate(small_flow, omega=1000.0, n_blades=50)

        history = solver.get_convergence_history()

        assert "iterations" in history
        assert "exit_angle" in history
        assert "blade_force" in history
        assert "rotation_angle" in history
        assert len(history["iterations"]) == 5


class TestModuleImports:
    """模組導入測試。"""

    def test_import_inverse_design_type(self) -> None:
        """測試導入 InverseDesignType。"""
        from multall_turbomachinery_design.multall.inverse_design import (
            InverseDesignType,
        )

        assert InverseDesignType is not None

    def test_import_inverse_design_solver(self) -> None:
        """測試導入 InverseDesignSolver。"""
        from multall_turbomachinery_design.multall.inverse_design import (
            InverseDesignSolver,
        )

        assert InverseDesignSolver is not None

    def test_import_blade_redesigner(self) -> None:
        """測試導入 BladeRedesigner。"""
        from multall_turbomachinery_design.multall.inverse_design import (
            BladeRedesigner,
        )

        assert BladeRedesigner is not None
