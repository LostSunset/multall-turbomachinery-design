# -*- coding: utf-8 -*-
"""MULTALL 基礎測試。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multall_turbomachinery_design.multall import (
    BladeRowGeometry,
    ExitBoundary,
    FlowField,
    GasProperties,
    GasType,
    Grid3D,
    GridParameters,
    InletBoundary,
    MixingPlaneParameters,
    MultallConfig,
    MultallSolver,
    SolverParameters,
    TimeStepType,
    ViscousModel,
    ViscousParameters,
    create_air_calculator,
    create_combustion_gas_calculator,
    create_simple_turbine_solver,
)


class TestGasProperties:
    """測試氣體性質數據結構。"""

    def test_default_air_properties(self) -> None:
        """測試預設空氣性質。"""
        gas = GasProperties()

        assert gas.cp == pytest.approx(1005.0)
        assert gas.gamma == pytest.approx(1.4)
        assert gas.gas_type == GasType.PERFECT_GAS

    def test_derived_properties(self) -> None:
        """測試派生性質計算。"""
        gas = GasProperties(cp=1005.0, gamma=1.4)

        assert gas.cv == pytest.approx(1005.0 / 1.4, rel=0.01)
        assert gas.rgas == pytest.approx(287.0, rel=0.01)

    def test_variable_cp_gas(self) -> None:
        """測試變 CP 氣體。"""
        gas = GasProperties(
            gas_type=GasType.VARIABLE_CP,
            cp1=1272.5,
            cp2=0.2125,
            cp3=0.000015625,
            tref=1400.0,
            rgas=287.15,
        )

        assert gas.gas_type == GasType.VARIABLE_CP
        assert gas.cp1 == 1272.5


class TestGasCalculator:
    """測試氣體計算器。"""

    def test_air_calculator(self) -> None:
        """測試空氣計算器。"""
        calc = create_air_calculator()

        assert calc.gamma == pytest.approx(1.4)
        assert calc.cp == pytest.approx(1005.0)

    def test_enthalpy_from_temperature(self) -> None:
        """測試從溫度計算焓。"""
        calc = create_air_calculator()
        t = 300.0  # K

        h = calc.enthalpy_from_temperature(t)

        assert h == pytest.approx(1005.0 * 300.0)

    def test_temperature_from_enthalpy(self) -> None:
        """測試從焓計算溫度。"""
        calc = create_air_calculator()
        h = 300000.0  # J/kg

        t = calc.temperature_from_enthalpy(h)

        assert t == pytest.approx(300000.0 / 1005.0, rel=0.01)

    def test_speed_of_sound(self) -> None:
        """測試聲速計算。"""
        calc = create_air_calculator()
        t = 288.15  # K (標準大氣)

        a = calc.speed_of_sound(t)

        # 標準大氣聲速約 340 m/s
        assert a == pytest.approx(340.0, rel=0.02)

    def test_total_conditions(self) -> None:
        """測試總條件計算。"""
        calc = create_air_calculator()
        t_static = 288.15  # K
        v = 100.0  # m/s

        t_total = calc.total_temperature(t_static, v)

        # T0 = T + V^2/(2*Cp)
        expected = t_static + v**2 / (2 * calc.cp)
        assert t_total == pytest.approx(expected, rel=0.01)

    def test_static_from_total(self) -> None:
        """測試從總條件計算靜態條件。"""
        calc = create_air_calculator()
        po = 200000.0  # Pa
        to = 400.0  # K
        mach = 0.5

        p_static, t_static = calc.static_from_total(po, to, mach)

        # 檢查靜溫小於總溫
        assert t_static < to
        # 檢查靜壓小於總壓
        assert p_static < po

    def test_density(self) -> None:
        """測試密度計算。"""
        calc = create_air_calculator()
        p = 101325.0  # Pa
        t = 288.15  # K

        rho = calc.density(p, t)

        # 標準大氣密度約 1.225 kg/m³
        assert rho == pytest.approx(1.225, rel=0.02)

    def test_mach_from_velocity(self) -> None:
        """測試馬赫數計算。"""
        calc = create_air_calculator()
        v = 170.0  # m/s
        t = 288.15  # K

        mach = calc.mach_from_velocity(v, t)

        # M ≈ 170/340 ≈ 0.5
        assert mach == pytest.approx(0.5, rel=0.05)

    def test_isentropic_relations(self) -> None:
        """測試等熵關係。"""
        calc = create_air_calculator()
        mach = 0.5

        t_ratio, p_ratio, rho_ratio, a_ratio = calc.isentropic_relations(mach)

        # 檢查比值在合理範圍
        assert 0 < t_ratio < 1
        assert 0 < p_ratio < 1
        assert 0 < rho_ratio < 1
        assert a_ratio > 1

    def test_combustion_gas_calculator(self) -> None:
        """測試燃氣計算器。"""
        calc = create_combustion_gas_calculator()

        assert calc.gas.gas_type == GasType.VARIABLE_CP
        assert calc.gas.cp1 == 1272.5


class TestGridParameters:
    """測試網格參數。"""

    def test_default_parameters(self) -> None:
        """測試預設參數。"""
        grid = GridParameters()

        assert grid.im == 37
        assert grid.km == 11
        assert grid.ir == 3
        assert grid.jr == 3
        assert grid.kr == 3

    def test_custom_parameters(self) -> None:
        """測試自定義參數。"""
        grid = GridParameters(im=17, jm=50, km=9)

        assert grid.im == 17
        assert grid.jm == 50
        assert grid.km == 9


class TestSolverParameters:
    """測試求解器參數。"""

    def test_default_parameters(self) -> None:
        """測試預設參數。"""
        solver = SolverParameters()

        assert solver.cfl == pytest.approx(0.4)
        assert solver.max_steps == 5000
        assert solver.time_step_type == TimeStepType.SCREE

    def test_custom_parameters(self) -> None:
        """測試自定義參數。"""
        solver = SolverParameters(
            cfl=0.7,
            max_steps=10000,
            convergence_limit=0.001,
        )

        assert solver.cfl == pytest.approx(0.7)
        assert solver.max_steps == 10000
        assert solver.convergence_limit == pytest.approx(0.001)


class TestViscousParameters:
    """測試黏性參數。"""

    def test_default_parameters(self) -> None:
        """測試預設參數。"""
        viscous = ViscousParameters()

        assert viscous.model == ViscousModel.MIXING_LENGTH
        assert viscous.reynolds == pytest.approx(500000.0)

    def test_spalart_allmaras(self) -> None:
        """測試 Spalart-Allmaras 模型。"""
        viscous = ViscousParameters(model=ViscousModel.SPALART_ALLMARAS)

        assert viscous.model == ViscousModel.SPALART_ALLMARAS


class TestBoundaryConditions:
    """測試邊界條件。"""

    def test_inlet_boundary(self) -> None:
        """測試進口邊界。"""
        inlet = InletBoundary(
            use_total_pressure=True,
            po=[200000.0] * 5,
            to=[400.0] * 5,
        )

        assert inlet.use_total_pressure
        assert len(inlet.po) == 5
        assert inlet.po[0] == pytest.approx(200000.0)

    def test_exit_boundary(self) -> None:
        """測試出口邊界。"""
        exit_bc = ExitBoundary(
            use_static_pressure=True,
            pstatic_hub=100000.0,
            pstatic_tip=100000.0,
        )

        assert exit_bc.use_static_pressure
        assert exit_bc.pstatic_hub == pytest.approx(100000.0)


class TestBladeRowGeometry:
    """測試葉片排幾何。"""

    def test_rotor(self) -> None:
        """測試轉子。"""
        rotor = BladeRowGeometry(
            row_number=1,
            row_type="R",
            n_blades=30,
            rpm=10000.0,
        )

        assert rotor.row_type == "R"
        assert rotor.n_blades == 30
        assert rotor.rpm == pytest.approx(10000.0)

    def test_stator(self) -> None:
        """測試定子。"""
        stator = BladeRowGeometry(
            row_number=2,
            row_type="S",
            n_blades=40,
            rpm=0.0,
        )

        assert stator.row_type == "S"
        assert stator.rpm == 0.0


class TestFlowField:
    """測試流場數據結構。"""

    def test_initialization(self) -> None:
        """測試初始化。"""
        flow = FlowField(im=10, jm=20, km=5)
        flow.initialize()

        assert flow.rho.shape == (10, 20, 5)
        assert flow.vx.shape == (10, 20, 5)
        assert flow.p.shape == (10, 20, 5)


class TestGrid3D:
    """測試 3D 網格數據結構。"""

    def test_initialization(self) -> None:
        """測試初始化。"""
        grid = Grid3D(im=10, jm=20, km=5)
        grid.initialize()

        assert grid.x.shape == (20, 5)
        assert grid.r.shape == (20, 5)
        assert grid.theta.shape == (10, 20, 5)


class TestMultallConfig:
    """測試 MULTALL 配置。"""

    def test_default_config(self) -> None:
        """測試預設配置。"""
        config = MultallConfig()

        assert config.gas.gamma == pytest.approx(1.4)
        assert config.grid.im == 37
        assert config.solver.cfl == pytest.approx(0.4)

    def test_custom_config(self) -> None:
        """測試自定義配置。"""
        config = MultallConfig(
            title="Test Turbine",
            nrows=2,
            blade_rows=[
                BladeRowGeometry(row_number=1, row_type="R", n_blades=30, rpm=10000),
                BladeRowGeometry(row_number=2, row_type="S", n_blades=40, rpm=0),
            ],
        )

        assert config.title == "Test Turbine"
        assert config.nrows == 2
        assert len(config.blade_rows) == 2


class TestMultallSolver:
    """測試 MULTALL 求解器。"""

    def test_solver_initialization(self) -> None:
        """測試求解器初始化。"""
        config = MultallConfig(
            grid=GridParameters(im=5, jm=10, km=3),
            inlet=InletBoundary(po=[200000.0] * 3, to=[400.0] * 3),
        )
        solver = MultallSolver(config)

        assert solver.config is config
        assert solver.flow is None

    def test_initialize_grid(self) -> None:
        """測試網格初始化。"""
        config = MultallConfig(
            grid=GridParameters(im=5, jm=10, km=3),
            nrows=1,
        )
        solver = MultallSolver(config)

        solver.initialize_grid()

        assert solver.grid is not None
        assert solver.grid.im == 5
        assert solver.grid.km == 3

    def test_initialize_flow(self) -> None:
        """測試流場初始化。"""
        config = MultallConfig(
            grid=GridParameters(im=5, jm=10, km=3),
            nrows=1,
            inlet=InletBoundary(
                po=[200000.0] * 3,
                to=[400.0] * 3,
            ),
        )
        solver = MultallSolver(config)

        solver.initialize_flow()

        assert solver.flow is not None
        assert solver.flow.rho.shape[0] == 5
        assert solver.flow.rho.shape[2] == 3

    def test_progress_callback(self) -> None:
        """測試進度回調。"""
        config = MultallConfig(
            grid=GridParameters(im=5, jm=10, km=3),
            solver=SolverParameters(max_steps=10),
        )
        solver = MultallSolver(config)

        callback_calls: list[tuple[int, float, float]] = []

        def callback(step: int, residual: float, mass_flow: float) -> None:
            callback_calls.append((step, residual, mass_flow))

        solver.set_progress_callback(callback)
        solver.initialize_flow()

        # 執行幾步
        for _ in range(3):
            solver._time_step()

        # 回調應該沒有被調用（因為我們直接調用 _time_step）
        # 完整測試需要調用 solve()


class TestCreateSimpleTurbineSolver:
    """測試便捷函數。"""

    def test_create_solver(self) -> None:
        """測試創建簡單渦輪求解器。"""
        solver = create_simple_turbine_solver(
            inlet_po=200000.0,
            inlet_to=400.0,
            exit_ps=100000.0,
            rpm=10000.0,
            n_blades=30,
        )

        assert isinstance(solver, MultallSolver)
        assert solver.config.nrows == 1
        assert len(solver.config.blade_rows) == 1
        assert solver.config.blade_rows[0].rpm == pytest.approx(10000.0)

    def test_solver_can_initialize(self) -> None:
        """測試求解器可以初始化。"""
        solver = create_simple_turbine_solver()

        solver.initialize_flow()

        assert solver.flow is not None
        assert solver.grid is not None


class TestMultallSolverOutput:
    """測試輸出功能。"""

    def test_run_with_output(self) -> None:
        """測試運行並輸出。"""
        solver = create_simple_turbine_solver()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = solver.run(output_dir=tmpdir)

            assert "converged" in result
            assert "steps" in result
            assert "output_dir" in result
            assert (Path(tmpdir) / "results.out").exists()


class TestMixingPlaneParameters:
    """測試混合平面參數。"""

    def test_default_parameters(self) -> None:
        """測試預設參數。"""
        mp = MixingPlaneParameters()

        assert mp.enabled
        assert mp.rfmix == pytest.approx(0.025)

    def test_disabled(self) -> None:
        """測試禁用混合平面。"""
        mp = MixingPlaneParameters(enabled=False)

        assert not mp.enabled
