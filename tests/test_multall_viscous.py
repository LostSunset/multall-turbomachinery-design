# -*- coding: utf-8 -*-
"""MULTALL 黏性模型測試。"""

from __future__ import annotations

import numpy as np
import pytest

from multall_turbomachinery_design.multall import (
    FlowField,
    GasProperties,
    GasType,
    Grid3D,
    MixingLengthModel,
    SpalartAllmarasModel,
    ViscousFluxCalculator,
    ViscousModel,
    WallDistanceCalculator,
)


class TestWallDistanceCalculator:
    """WallDistanceCalculator 類測試。"""

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        return Grid3D(im=5, jm=10, km=9)

    def test_wall_distance_init(self, small_grid: Grid3D) -> None:
        """測試壁面距離計算器初始化。"""
        calc = WallDistanceCalculator(small_grid)
        assert calc.grid is small_grid

    def test_wall_distance_compute_shape(self, small_grid: Grid3D) -> None:
        """測試壁面距離計算形狀。"""
        calc = WallDistanceCalculator(small_grid)
        d = calc.compute()

        assert d.shape == (5, 10, 9)

    def test_wall_distance_zero_at_walls(self, small_grid: Grid3D) -> None:
        """測試壁面距離在壁面為零。"""
        calc = WallDistanceCalculator(small_grid)
        d = calc.compute()

        # K=0 (HUB) 和 K=KM-1 (CASING) 處距離應為零
        assert np.allclose(d[:, :, 0], 0.0)
        assert np.allclose(d[:, :, -1], 0.0)

    def test_wall_distance_positive_interior(self, small_grid: Grid3D) -> None:
        """測試內部壁面距離為正。"""
        calc = WallDistanceCalculator(small_grid)
        d = calc.compute()

        # 內部點距離應為正
        assert np.all(d[:, :, 1:-1] > 0)

    def test_wall_distance_maximum_at_center(self, small_grid: Grid3D) -> None:
        """測試壁面距離在中心最大。"""
        calc = WallDistanceCalculator(small_grid)
        d = calc.compute()

        # 在 K=4（中心）處距離應最大
        center_k = small_grid.km // 2
        assert np.all(d[:, :, center_k] >= d[:, :, 1])
        assert np.all(d[:, :, center_k] >= d[:, :, -2])

    def test_wall_distance_caching(self, small_grid: Grid3D) -> None:
        """測試壁面距離緩存。"""
        calc = WallDistanceCalculator(small_grid)
        d1 = calc.compute()
        d2 = calc.compute()

        # 應該是同一個數組（緩存）
        assert d1 is d2


class TestMixingLengthModel:
    """MixingLengthModel 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 9
        flow = FlowField(im=im, jm=jm, km=km)

        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)
        flow.t_static = np.full((im, jm, km), 300.0)
        flow.vx = np.full((im, jm, km), 100.0)
        flow.vr = np.zeros((im, jm, km))
        flow.vt = np.full((im, jm, km), 50.0)

        # 添加邊界層效應（壁面速度為零）
        flow.vx[:, :, 0] = 0.0
        flow.vx[:, :, -1] = 0.0
        flow.vt[:, :, 0] = 0.0
        flow.vt[:, :, -1] = 0.0

        return flow

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        return Grid3D(im=5, jm=10, km=9)

    def test_mixing_length_init(self, gas: GasProperties) -> None:
        """測試混合長度模型初始化。"""
        model = MixingLengthModel(gas)
        assert model.gas is gas

    def test_mixing_length_constants(self, gas: GasProperties) -> None:
        """測試混合長度模型常數。"""
        model = MixingLengthModel(gas)
        assert model.KAPPA == pytest.approx(0.41)
        assert model.A_PLUS == pytest.approx(26.0)

    def test_compute_mixing_length_shape(self, gas: GasProperties) -> None:
        """測試混合長度計算形狀。"""
        model = MixingLengthModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        y_plus = np.ones((5, 10, 9)) * 100.0

        l_m = model.compute_mixing_length(wall_distance, y_plus)

        assert l_m.shape == (5, 10, 9)

    def test_compute_mixing_length_zero_at_wall(self, gas: GasProperties) -> None:
        """測試混合長度在壁面為零。"""
        model = MixingLengthModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        y_plus = np.ones((5, 10, 9)) * 100.0
        y_plus[:, :, 0] = 0.0

        l_m = model.compute_mixing_length(wall_distance, y_plus)

        assert np.allclose(l_m[:, :, 0], 0.0)

    def test_compute_eddy_viscosity_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試渦黏性計算形狀。"""
        model = MixingLengthModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        mu_t = model.compute_eddy_viscosity(small_flow, small_grid, wall_distance)

        assert mu_t.shape == (5, 10, 9)

    def test_compute_eddy_viscosity_positive(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試渦黏性非負。"""
        model = MixingLengthModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        mu_t = model.compute_eddy_viscosity(small_flow, small_grid, wall_distance)

        assert np.all(mu_t >= 0)


class TestSpalartAllmarasModel:
    """SpalartAllmarasModel 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 9
        flow = FlowField(im=im, jm=jm, km=km)

        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)
        flow.t_static = np.full((im, jm, km), 300.0)
        flow.vx = np.full((im, jm, km), 100.0)
        flow.vr = np.zeros((im, jm, km))
        flow.vt = np.full((im, jm, km), 50.0)

        return flow

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        return Grid3D(im=5, jm=10, km=9)

    def test_sa_model_init(self, gas: GasProperties) -> None:
        """測試 SA 模型初始化。"""
        model = SpalartAllmarasModel(gas)
        assert model.gas is gas

    def test_sa_model_constants(self, gas: GasProperties) -> None:
        """測試 SA 模型常數。"""
        model = SpalartAllmarasModel(gas)
        assert model.CB1 == pytest.approx(0.1355)
        assert model.KAPPA == pytest.approx(0.41)
        assert model.CV1 == pytest.approx(7.1)

    def test_sa_initialize(
        self,
        gas: GasProperties,
        small_flow: FlowField,
    ) -> None:
        """測試 SA 模型初始化。"""
        model = SpalartAllmarasModel(gas)
        model.initialize(small_flow)

        assert model._nu_tilde is not None
        assert model._nu_tilde.shape == (5, 10, 9)
        assert np.all(model._nu_tilde > 0)

    def test_sa_compute_eddy_viscosity_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 SA 渦黏性計算形狀。"""
        model = SpalartAllmarasModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        mu_t = model.compute_eddy_viscosity(small_flow, small_grid, wall_distance)

        assert mu_t.shape == (5, 10, 9)

    def test_sa_compute_eddy_viscosity_positive(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 SA 渦黏性非負。"""
        model = SpalartAllmarasModel(gas)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        mu_t = model.compute_eddy_viscosity(small_flow, small_grid, wall_distance)

        assert np.all(mu_t >= 0)

    def test_sa_compute_source_terms_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 SA 源項形狀。"""
        model = SpalartAllmarasModel(gas)
        model.initialize(small_flow)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        source = model.compute_source_terms(small_flow, small_grid, wall_distance)

        assert source.shape == (5, 10, 9)

    def test_sa_update(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試 SA 更新。"""
        model = SpalartAllmarasModel(gas)
        model.initialize(small_flow)

        wall_distance = np.ones((5, 10, 9)) * 0.01
        wall_distance[:, :, 0] = 0.0
        wall_distance[:, :, -1] = 0.0

        nu_tilde_before = model._nu_tilde.copy()

        dt = np.full((5, 10, 9), 0.0001)
        model.update(small_flow, small_grid, wall_distance, dt)

        # nu_tilde 應該改變
        assert not np.allclose(model._nu_tilde, nu_tilde_before)

        # 壁面值應為零
        assert np.allclose(model._nu_tilde[:, :, 0], 0.0)
        assert np.allclose(model._nu_tilde[:, :, -1], 0.0)


class TestViscousFluxCalculator:
    """ViscousFluxCalculator 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 10, 9
        flow = FlowField(im=im, jm=jm, km=km)

        flow.rho = np.full((im, jm, km), 1.2)
        flow.p = np.full((im, jm, km), 101325.0)
        flow.t_static = np.full((im, jm, km), 300.0)
        flow.vx = np.full((im, jm, km), 100.0)
        flow.vr = np.zeros((im, jm, km))
        flow.vt = np.full((im, jm, km), 50.0)

        return flow

    @pytest.fixture
    def small_grid(self) -> Grid3D:
        """創建小型網格。"""
        grid = Grid3D(im=5, jm=10, km=9)
        grid.initialize()
        return grid

    def test_viscous_flux_calc_init_mixing_length(self, gas: GasProperties) -> None:
        """測試混合長度模型黏性通量計算器初始化。"""
        calc = ViscousFluxCalculator(gas, ViscousModel.MIXING_LENGTH)
        assert calc.viscous_model == ViscousModel.MIXING_LENGTH
        assert isinstance(calc.turbulence_model, MixingLengthModel)

    def test_viscous_flux_calc_init_sa(self, gas: GasProperties) -> None:
        """測試 SA 模型黏性通量計算器初始化。"""
        calc = ViscousFluxCalculator(gas, ViscousModel.SPALART_ALLMARAS)
        assert calc.viscous_model == ViscousModel.SPALART_ALLMARAS
        assert isinstance(calc.turbulence_model, SpalartAllmarasModel)

    def test_viscous_flux_calc_init_inviscid(self, gas: GasProperties) -> None:
        """測試無黏性通量計算器初始化。"""
        calc = ViscousFluxCalculator(gas, ViscousModel.INVISCID)
        assert calc.viscous_model == ViscousModel.INVISCID
        assert calc.turbulence_model is None

    def test_compute_viscous_flux_inviscid(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試無黏性通量計算。"""
        calc = ViscousFluxCalculator(gas, ViscousModel.INVISCID)
        tau_xx, tau_rr, tau_tt, energy, mu_t = calc.compute_viscous_flux(small_flow, small_grid)

        # 無黏性應該全為零
        assert np.allclose(tau_xx, 0.0)
        assert np.allclose(tau_rr, 0.0)
        assert np.allclose(tau_tt, 0.0)
        assert np.allclose(mu_t, 0.0)

    def test_compute_viscous_flux_shape(
        self,
        gas: GasProperties,
        small_flow: FlowField,
        small_grid: Grid3D,
    ) -> None:
        """測試黏性通量形狀。"""
        calc = ViscousFluxCalculator(gas, ViscousModel.MIXING_LENGTH)
        tau_xx, tau_rr, tau_tt, energy, mu_t = calc.compute_viscous_flux(small_flow, small_grid)

        assert tau_xx.shape == (5, 10, 9)
        assert tau_rr.shape == (5, 10, 9)
        assert tau_tt.shape == (5, 10, 9)
        assert energy.shape == (5, 10, 9)
        assert mu_t.shape == (5, 10, 9)


class TestModuleImports:
    """模組導入測試。"""

    def test_import_wall_distance_calculator(self) -> None:
        """測試導入 WallDistanceCalculator。"""
        from multall_turbomachinery_design.multall import WallDistanceCalculator

        assert WallDistanceCalculator is not None

    def test_import_mixing_length_model(self) -> None:
        """測試導入 MixingLengthModel。"""
        from multall_turbomachinery_design.multall import MixingLengthModel

        assert MixingLengthModel is not None

    def test_import_spalart_allmaras_model(self) -> None:
        """測試導入 SpalartAllmarasModel。"""
        from multall_turbomachinery_design.multall import SpalartAllmarasModel

        assert SpalartAllmarasModel is not None

    def test_import_viscous_flux_calculator(self) -> None:
        """測試導入 ViscousFluxCalculator。"""
        from multall_turbomachinery_design.multall import ViscousFluxCalculator

        assert ViscousFluxCalculator is not None
