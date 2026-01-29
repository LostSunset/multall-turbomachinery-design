# -*- coding: utf-8 -*-
"""MULTALL 通量計算和時間步進模組測試。"""

from __future__ import annotations

import numpy as np
import pytest

from multall_turbomachinery_design.multall import (
    ArtificialViscosity,
    ConvergenceMonitor,
    FlowField,
    FluxCalculator,
    GasProperties,
    GasType,
    Grid3D,
    SolverParameters,
    TimeStepMethod,
    TimeStepper,
)


class TestFluxCalculator:
    """FluxCalculator 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def small_flow(self, gas: GasProperties) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 3
        flow = FlowField(im=im, jm=jm, km=km)

        # 初始化流場
        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)
        flow.t_static = np.full((im, jm, km), 300.0)
        flow.vx = np.full((im, jm, km), 100.0)
        flow.vr = np.zeros((im, jm, km))
        flow.vt = np.full((im, jm, km), 50.0)

        # 計算總焓
        cp = gas.gamma * gas.rgas / (gas.gamma - 1.0)
        v_sq = flow.vx**2 + flow.vr**2 + flow.vt**2
        flow.ho = cp * flow.t_static + 0.5 * v_sq

        # 守恆變量
        flow.ro = flow.rho.copy()
        flow.rovx = flow.rho * flow.vx
        flow.rovr = flow.rho * flow.vr
        flow.rorvt = flow.rho * flow.vt  # 簡化，未乘 r
        flow.roe = flow.rho * (cp * flow.t_static / gas.gamma + 0.5 * v_sq)

        return flow

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        im, jm, km = 5, 10, 3
        return Grid3D(im=im, jm=jm, km=km)

    def test_flux_calculator_init(self, gas: GasProperties) -> None:
        """測試 FluxCalculator 初始化。"""
        calc = FluxCalculator(gas)
        assert calc.gas == gas
        assert calc.gas_calc is not None

    def test_convective_flux_x_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 X 方向通量形狀。"""
        calc = FluxCalculator(gas)
        flux = calc.compute_convective_flux_x(small_flow, small_grid)

        assert len(flux) == 5
        # 面上的通量，J 方向 +1
        assert flux[0].shape == (5, 11, 3)
        assert flux[1].shape == (5, 11, 3)
        assert flux[2].shape == (5, 11, 3)
        assert flux[3].shape == (5, 11, 3)
        assert flux[4].shape == (5, 11, 3)

    def test_convective_flux_x_mass_conservation(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試質量通量連續性。"""
        calc = FluxCalculator(gas)
        flux_mass, _, _, _, _ = calc.compute_convective_flux_x(small_flow, small_grid)

        # 均勻流場應該有相對一致的通量
        inner_flux = flux_mass[:, 1:-1, :]
        assert np.allclose(inner_flux, inner_flux.mean(), rtol=0.1)

    def test_convective_flux_theta_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 θ 方向通量形狀。"""
        calc = FluxCalculator(gas)
        flux = calc.compute_convective_flux_theta(small_flow, small_grid)

        assert len(flux) == 5
        # I 方向 +1
        assert flux[0].shape == (6, 10, 3)

    def test_convective_flux_r_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 R 方向通量形狀。"""
        calc = FluxCalculator(gas)
        flux = calc.compute_convective_flux_r(small_flow, small_grid)

        assert len(flux) == 5
        # K 方向 +1
        assert flux[0].shape == (5, 10, 4)

    def test_convective_flux_r_wall_boundary(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 R 方向壁面邊界。"""
        calc = FluxCalculator(gas)
        flux_mass, _, _, _, _ = calc.compute_convective_flux_r(small_flow, small_grid)

        # 壁面質量通量應為零
        assert np.allclose(flux_mass[:, :, 0], 0.0)
        assert np.allclose(flux_mass[:, :, -1], 0.0)

    def test_compute_residual_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試殘差形狀。"""
        calc = FluxCalculator(gas)
        flux_x = calc.compute_convective_flux_x(small_flow, small_grid)
        flux_theta = calc.compute_convective_flux_theta(small_flow, small_grid)
        flux_r = calc.compute_convective_flux_r(small_flow, small_grid)

        residual = calc.compute_residual(small_flow, small_grid, flux_x, flux_theta, flux_r)

        assert len(residual) == 5
        for res in residual:
            assert res.shape == (5, 10, 3)


class TestArtificialViscosity:
    """ArtificialViscosity 類測試。"""

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 3
        flow = FlowField(im=im, jm=jm, km=km)

        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)

        flow.ro = flow.rho.copy()
        flow.rovx = flow.rho * 100.0
        flow.rovr = np.zeros((im, jm, km))
        flow.rorvt = flow.rho * 50.0
        flow.roe = flow.rho * 250000.0

        return flow

    def test_artificial_viscosity_init(self) -> None:
        """測試 ArtificialViscosity 初始化。"""
        av = ArtificialViscosity(sf_2nd=0.01, sf_4th=0.5)
        assert av.sf_2nd == 0.01
        assert av.sf_4th == 0.5

    def test_pressure_sensor_shape(self, small_flow: FlowField) -> None:
        """測試壓力感測器形狀。"""
        av = ArtificialViscosity()
        sensor = av.compute_pressure_sensor(small_flow.p)
        assert sensor.shape == small_flow.p.shape

    def test_pressure_sensor_uniform_flow(self, small_flow: FlowField) -> None:
        """測試均勻流場壓力感測器。"""
        av = ArtificialViscosity()
        sensor = av.compute_pressure_sensor(small_flow.p)

        # 均勻壓力，感測器應接近零
        assert np.allclose(sensor[:, 1:-1, :], 0.0, atol=1e-10)

    def test_pressure_sensor_with_shock(self) -> None:
        """測試含激波壓力感測器。"""
        im, jm, km = 5, 10, 3
        p = np.full((im, jm, km), 100000.0)

        # 在 j=5 處添加壓力跳躍（模擬激波）
        p[:, 5:, :] = 200000.0

        av = ArtificialViscosity()
        sensor = av.compute_pressure_sensor(p)

        # 激波位置感測器應較大
        assert sensor[:, 5, :].max() > sensor[:, 2, :].max()

    def test_artificial_dissipation_shape(self, small_flow: FlowField) -> None:
        """測試人工耗散形狀。"""
        av = ArtificialViscosity()
        diss = av.compute_artificial_dissipation(small_flow, direction="x")

        assert len(diss) == 5
        for d in diss:
            assert d.shape == (5, 10, 3)


class TestTimeStepper:
    """TimeStepper 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def solver_params(self) -> SolverParameters:
        """創建求解器參數。"""
        return SolverParameters(
            max_steps=1000,
            convergence_limit=0.001,
            cfl=0.5,
        )

    @pytest.fixture
    def small_flow(self, gas: GasProperties) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 3
        flow = FlowField(im=im, jm=jm, km=km)

        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)
        flow.t_static = np.full((im, jm, km), 300.0)
        flow.vx = np.full((im, jm, km), 100.0)
        flow.vr = np.zeros((im, jm, km))
        flow.vt = np.full((im, jm, km), 50.0)

        cp = gas.gamma * gas.rgas / (gas.gamma - 1.0)
        v_sq = flow.vx**2 + flow.vr**2 + flow.vt**2

        flow.ro = flow.rho.copy()
        flow.rovx = flow.rho * flow.vx
        flow.rovr = flow.rho * flow.vr
        flow.rorvt = flow.rho * flow.vt
        flow.roe = flow.rho * (cp * flow.t_static / gas.gamma + 0.5 * v_sq)

        return flow

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        return Grid3D(im=5, jm=10, km=3)

    def test_time_stepper_init(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
    ) -> None:
        """測試 TimeStepper 初始化。"""
        stepper = TimeStepper(gas, solver_params)
        assert stepper.gas == gas
        assert stepper.params == solver_params

    def test_local_time_step_shape(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試局部時間步長形狀。"""
        stepper = TimeStepper(gas, solver_params)
        dt = stepper.compute_local_time_step(small_flow, small_grid)

        assert dt.shape == (5, 10, 3)
        assert np.all(dt > 0)

    def test_local_time_step_positive(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試時間步長正值。"""
        stepper = TimeStepper(gas, solver_params)
        dt = stepper.compute_local_time_step(small_flow, small_grid)

        assert np.all(dt > 0)
        assert np.all(dt < 1.0)  # 應該是很小的值

    def test_euler_step(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
        small_flow: FlowField,
    ) -> None:
        """測試 Euler 步進。"""
        stepper = TimeStepper(gas, solver_params)

        # 保存初始狀態
        ro_init = small_flow.ro.copy()

        # 創建殘差
        residual = (
            np.ones_like(small_flow.ro),
            np.zeros_like(small_flow.rovx),
            np.zeros_like(small_flow.rovr),
            np.zeros_like(small_flow.rorvt),
            np.zeros_like(small_flow.roe),
        )

        dt = np.full_like(small_flow.ro, 0.0001)

        stepper.euler_step(small_flow, residual, dt)

        # 密度應該減少（因為正殘差）
        assert np.all(small_flow.ro < ro_init)

    def test_scree_step(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
        small_flow: FlowField,
    ) -> None:
        """測試 SCREE 步進。"""
        stepper = TimeStepper(gas, solver_params)

        ro_init = small_flow.ro.copy()

        # 使用較大的殘差使變化更明顯
        residual = (
            np.ones_like(small_flow.ro) * 100.0,
            np.zeros_like(small_flow.rovx),
            np.zeros_like(small_flow.rovr),
            np.zeros_like(small_flow.rorvt),
            np.zeros_like(small_flow.roe),
        )

        dt = np.full_like(small_flow.ro, 0.001)

        stepper.scree_step(small_flow, residual, dt)

        # 密度應該減少（因為正殘差）
        assert np.all(small_flow.ro < ro_init)

    def test_time_step_method_enum(self) -> None:
        """測試時間步進方法枚舉。"""
        assert TimeStepMethod.EULER == 1
        assert TimeStepMethod.RK2 == 2
        assert TimeStepMethod.RK4 == 4
        assert TimeStepMethod.SCREE == 3


class TestConvergenceMonitor:
    """ConvergenceMonitor 類測試。"""

    def test_monitor_init(self) -> None:
        """測試監視器初始化。"""
        monitor = ConvergenceMonitor(convergence_limit=0.01, history_size=50)
        assert monitor.convergence_limit == 0.01
        assert monitor.history_size == 50

    def test_add_residual(self) -> None:
        """測試添加殘差。"""
        monitor = ConvergenceMonitor()

        for i in range(10):
            monitor.add_residual(1.0 / (i + 1))

        assert len(monitor.residual_history) == 10
        assert monitor.current_residual == pytest.approx(0.1)

    def test_add_mass_flow(self) -> None:
        """測試添加質量流量。"""
        monitor = ConvergenceMonitor()

        for i in range(5):
            monitor.add_mass_flow(10.0 + i * 0.1)

        assert len(monitor.mass_flow_history) == 5

    def test_history_size_limit(self) -> None:
        """測試歷史大小限制。"""
        monitor = ConvergenceMonitor(history_size=10)

        for i in range(20):
            monitor.add_residual(float(i))

        assert len(monitor.residual_history) == 10
        assert monitor.residual_history[0] == 10.0  # 前 10 個被移除

    def test_compute_l2_residual(self) -> None:
        """測試 L2 殘差計算。"""
        monitor = ConvergenceMonitor()

        residual = (
            np.ones((3, 3, 3)),
            np.ones((3, 3, 3)),
        )

        l2 = monitor.compute_l2_residual(residual)

        # sqrt(54 / 54) = 1.0
        assert l2 == pytest.approx(1.0)

    def test_is_converged_absolute(self) -> None:
        """測試絕對收斂判斷。"""
        monitor = ConvergenceMonitor(convergence_limit=0.01)

        monitor.add_residual(0.001)

        assert monitor.is_converged()

    def test_is_converged_relative(self) -> None:
        """測試相對收斂判斷。"""
        monitor = ConvergenceMonitor(convergence_limit=0.01)

        # 初始殘差
        monitor.add_residual(1000.0)

        # 下降 3 個數量級
        monitor.add_residual(0.5)

        assert monitor.is_converged()

    def test_is_not_converged(self) -> None:
        """測試未收斂判斷。"""
        monitor = ConvergenceMonitor(convergence_limit=0.001)

        monitor.add_residual(1.0)
        monitor.add_residual(0.5)

        assert not monitor.is_converged()

    def test_is_stalled(self) -> None:
        """測試停滯判斷。"""
        monitor = ConvergenceMonitor()

        # 添加幾乎相同的殘差
        for _ in range(60):
            monitor.add_residual(0.1)

        assert monitor.is_stalled(window=50)

    def test_is_not_stalled(self) -> None:
        """測試未停滯判斷。"""
        monitor = ConvergenceMonitor()

        # 添加遞減殘差
        for i in range(60):
            monitor.add_residual(1.0 / (i + 1))

        assert not monitor.is_stalled(window=50)

    def test_convergence_rate_decreasing(self) -> None:
        """測試遞減殘差收斂率。"""
        monitor = ConvergenceMonitor()

        # 添加指數遞減殘差
        for i in range(30):
            monitor.add_residual(np.exp(-0.1 * i))

        rate = monitor.get_convergence_rate(window=20)

        # 應該是負值（表示收斂）
        assert rate < 0

    def test_normalized_residual(self) -> None:
        """測試歸一化殘差。"""
        monitor = ConvergenceMonitor()

        monitor.add_residual(100.0)
        monitor.add_residual(10.0)

        assert monitor.normalized_residual == pytest.approx(0.1)

    def test_empty_monitor(self) -> None:
        """測試空監視器。"""
        monitor = ConvergenceMonitor()

        assert not monitor.is_converged()
        assert monitor.current_residual == 1.0
        assert monitor.normalized_residual == 1.0


class TestModuleImports:
    """模組導入測試。"""

    def test_import_flux_calculator(self) -> None:
        """測試導入 FluxCalculator。"""
        from multall_turbomachinery_design.multall import FluxCalculator

        assert FluxCalculator is not None

    def test_import_artificial_viscosity(self) -> None:
        """測試導入 ArtificialViscosity。"""
        from multall_turbomachinery_design.multall import ArtificialViscosity

        assert ArtificialViscosity is not None

    def test_import_time_stepper(self) -> None:
        """測試導入 TimeStepper。"""
        from multall_turbomachinery_design.multall import TimeStepper

        assert TimeStepper is not None

    def test_import_convergence_monitor(self) -> None:
        """測試導入 ConvergenceMonitor。"""
        from multall_turbomachinery_design.multall import ConvergenceMonitor

        assert ConvergenceMonitor is not None

    def test_import_time_step_method(self) -> None:
        """測試導入 TimeStepMethod。"""
        from multall_turbomachinery_design.multall import TimeStepMethod

        assert TimeStepMethod is not None
