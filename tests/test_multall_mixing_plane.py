# -*- coding: utf-8 -*-
"""MULTALL 混合平面模型測試。"""

from __future__ import annotations

import numpy as np
import pytest

from multall_turbomachinery_design.multall import (
    FlowField,
    GasProperties,
    GasType,
    MixingPlaneInterface,
    MixingPlaneModel,
    MixingPlaneParameters,
    MixingPlaneType,
    NonReflectingBoundary,
)


class TestMixingPlaneInterface:
    """MixingPlaneInterface 類測試。"""

    def test_interface_creation(self) -> None:
        """測試交界面創建。"""
        interface = MixingPlaneInterface(j_upstream=10, j_downstream=11)

        assert interface.j_upstream == 10
        assert interface.j_downstream == 11

    def test_interface_default_arrays(self) -> None:
        """測試交界面默認數組。"""
        interface = MixingPlaneInterface()

        assert interface.rho_avg.size == 0
        assert interface.vx_avg.size == 0
        assert interface.p_avg.size == 0


class TestMixingPlaneModel:
    """MixingPlaneModel 類測試。"""

    @pytest.fixture
    def gas(self) -> GasProperties:
        """創建氣體性質。"""
        return GasProperties(
            gamma=1.4,
            cp=1005.0,
            gas_type=GasType.PERFECT_GAS,
        )

    @pytest.fixture
    def params(self) -> MixingPlaneParameters:
        """創建混合平面參數。"""
        return MixingPlaneParameters(enabled=True)

    @pytest.fixture
    def small_flow(self) -> FlowField:
        """創建小型流場。"""
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()  # 創建 3D 陣列

        # 初始化流場（帶周向變化）
        for i in range(im):
            theta = 2 * np.pi * i / im
            for j in range(jm):
                for k in range(km):
                    flow.rho[i, j, k] = 1.2 + 0.1 * np.sin(theta)
                    flow.vx[i, j, k] = 100.0 + 10.0 * np.cos(theta)
                    flow.vr[i, j, k] = 5.0 * np.sin(theta)
                    flow.vt[i, j, k] = 50.0 + 5.0 * np.cos(theta)
                    flow.p[i, j, k] = 101325.0 + 1000.0 * np.sin(theta)
                    flow.t_static[i, j, k] = 300.0 + 10.0 * np.cos(theta)

        return flow

    def test_model_init(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
    ) -> None:
        """測試混合平面模型初始化。"""
        model = MixingPlaneModel(gas, params)

        assert model.gas is gas
        assert model.params is params
        assert len(model.interfaces) == 0

    def test_add_interface(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
    ) -> None:
        """測試添加交界面。"""
        model = MixingPlaneModel(gas, params)
        interface = model.add_interface(j_upstream=10, j_downstream=11)

        assert len(model.interfaces) == 1
        assert interface.j_upstream == 10
        assert interface.j_downstream == 11

    def test_compute_circumferential_average_shape(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試周向平均計算形狀。"""
        model = MixingPlaneModel(gas, params)
        avg = model.compute_circumferential_average(small_flow, j_index=10)

        assert avg["rho"].shape == (9,)
        assert avg["vx"].shape == (9,)
        assert avg["vr"].shape == (9,)
        assert avg["vt"].shape == (9,)
        assert avg["p"].shape == (9,)
        assert avg["t"].shape == (9,)
        assert avg["po"].shape == (9,)
        assert avg["to"].shape == (9,)

    def test_compute_circumferential_average_values(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試周向平均計算值。"""
        model = MixingPlaneModel(gas, params)
        avg = model.compute_circumferential_average(small_flow, j_index=10)

        # 平均值應該接近基礎值（周向變化平均後接近常數）
        assert np.allclose(avg["rho"], 1.2, atol=0.05)
        assert np.allclose(avg["vx"], 100.0, atol=5.0)
        assert np.allclose(avg["vt"], 50.0, atol=3.0)

    def test_compute_mass_average_shape(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試質量平均計算形狀。"""
        model = MixingPlaneModel(gas, params)
        avg = model.compute_mass_average(small_flow, j_index=10)

        assert avg["rho"].shape == (9,)
        assert avg["vx"].shape == (9,)
        assert "ho" in avg

    def test_compute_flux_average_shape(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試通量平均計算形狀。"""
        model = MixingPlaneModel(gas, params)
        avg = model.compute_flux_average(small_flow, j_index=10)

        assert avg["rho"].shape == (9,)
        assert avg["vx"].shape == (9,)

    def test_apply_mixing_plane(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試應用混合平面邊界條件。"""
        model = MixingPlaneModel(gas, params)
        interface = model.add_interface(j_upstream=10, j_downstream=11)

        # 應用前下游入口有周向變化
        assert not np.allclose(
            small_flow.rho[:, 11, :],
            small_flow.rho[0, 11, :],
        )

        model.apply_mixing_plane(small_flow, interface)

        # 應用後下游入口周向應該均勻
        for i in range(1, small_flow.im):
            assert np.allclose(
                small_flow.rho[i, 11, :],
                small_flow.rho[0, 11, :],
            )

    def test_apply_mixing_plane_updates_interface(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試應用混合平面更新交界面數據。"""
        model = MixingPlaneModel(gas, params)
        interface = model.add_interface(j_upstream=10, j_downstream=11)

        model.apply_mixing_plane(small_flow, interface)

        # 交界面數據應該被更新
        assert interface.rho_avg.size == 9
        assert interface.vx_avg.size == 9

    def test_averaging_type_selection(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
    ) -> None:
        """測試平均類型選擇。"""
        model = MixingPlaneModel(gas, params)

        model.averaging_type = MixingPlaneType.MASS_AVERAGE
        assert model.averaging_type == MixingPlaneType.MASS_AVERAGE

        model.averaging_type = MixingPlaneType.FLUX_AVERAGE
        assert model.averaging_type == MixingPlaneType.FLUX_AVERAGE

    def test_compute_interface_mass_flow(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試計算交界面質量流量。"""
        model = MixingPlaneModel(gas, params)
        mass_flow = model.compute_interface_mass_flow(small_flow, j_index=10)

        assert mass_flow > 0

    def test_compute_interface_efficiency(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試計算交界面效率。"""
        model = MixingPlaneModel(gas, params)
        interface = model.add_interface(j_upstream=10, j_downstream=11)

        efficiency = model.compute_interface_efficiency(small_flow, interface)

        assert "total_pressure_loss" in efficiency
        assert "total_temperature_ratio" in efficiency
        assert "pressure_ratio" in efficiency
        assert "isentropic_efficiency" in efficiency

    def test_update_all_interfaces(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試更新所有交界面。"""
        model = MixingPlaneModel(gas, params)
        model.add_interface(j_upstream=5, j_downstream=6)
        model.add_interface(j_upstream=10, j_downstream=11)

        model.update_all_interfaces(small_flow)

        # 所有下游入口都應該周向均勻
        for i in range(1, small_flow.im):
            assert np.allclose(small_flow.rho[i, 6, :], small_flow.rho[0, 6, :])
            assert np.allclose(small_flow.rho[i, 11, :], small_flow.rho[0, 11, :])

    def test_apply_rotor_stator_interface(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
        small_flow: FlowField,
    ) -> None:
        """測試應用轉子-靜子交界面。"""
        model = MixingPlaneModel(gas, params)
        interface = model.add_interface(j_upstream=10, j_downstream=11)

        # 上游是轉子（omega=1000 rad/s），下游是靜子（omega=0）
        model.apply_rotor_stator_interface(
            small_flow, interface, omega_upstream=1000.0, omega_downstream=0.0
        )

        # 周向速度應該被轉換
        assert interface.vt_avg.size == 9


class TestMixingPlaneType:
    """MixingPlaneType 枚舉測試。"""

    def test_enum_values(self) -> None:
        """測試枚舉值。"""
        assert MixingPlaneType.CIRCUMFERENTIAL_AVERAGE == 1
        assert MixingPlaneType.FLUX_AVERAGE == 2
        assert MixingPlaneType.AREA_AVERAGE == 3
        assert MixingPlaneType.MASS_AVERAGE == 4


class TestNonReflectingBoundary:
    """NonReflectingBoundary 類測試。"""

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
        im, jm, km = 5, 20, 9
        flow = FlowField(im=im, jm=jm, km=km)
        flow.initialize()  # 創建 3D 陣列

        flow.rho[:] = 1.2
        flow.vx[:] = 100.0
        flow.vr[:] = 0.0
        flow.vt[:] = 50.0
        flow.p[:] = 101325.0
        flow.t_static[:] = 300.0

        return flow

    def test_nrbc_init(self, gas: GasProperties) -> None:
        """測試無反射邊界初始化。"""
        nrbc = NonReflectingBoundary(gas, relaxation_factor=0.2)

        assert nrbc.gas is gas
        assert nrbc.relaxation_factor == 0.2

    def test_apply_inlet_nrbc(
        self,
        gas: GasProperties,
        small_flow: FlowField,
    ) -> None:
        """測試應用入口無反射邊界。"""
        nrbc = NonReflectingBoundary(gas, relaxation_factor=0.1)

        km = small_flow.km
        target_po = np.full(km, 200000.0)
        target_to = np.full(km, 400.0)
        target_alpha = np.zeros(km)

        p_before = small_flow.p[:, 0, :].copy()

        nrbc.apply_inlet_nrbc(
            small_flow,
            j_index=0,
            target_po=target_po,
            target_to=target_to,
            target_alpha=target_alpha,
        )

        # 壓力應該改變（向目標值鬆弛）
        assert not np.allclose(small_flow.p[:, 0, :], p_before)

    def test_apply_exit_nrbc(
        self,
        gas: GasProperties,
        small_flow: FlowField,
    ) -> None:
        """測試應用出口無反射邊界。"""
        nrbc = NonReflectingBoundary(gas, relaxation_factor=0.1)

        km = small_flow.km
        target_p = np.full(km, 80000.0)

        p_before = small_flow.p[:, -1, :].copy()

        nrbc.apply_exit_nrbc(
            small_flow,
            j_index=-1,
            target_p=target_p,
        )

        # 壓力應該改變（向目標值鬆弛）
        assert not np.allclose(small_flow.p[:, -1, :], p_before)

    def test_nrbc_relaxation_factor(
        self,
        gas: GasProperties,
        small_flow: FlowField,
    ) -> None:
        """測試無反射邊界鬆弛因子影響。"""
        km = small_flow.km
        target_p = np.full(km, 80000.0)

        # 較小的鬆弛因子
        nrbc_slow = NonReflectingBoundary(gas, relaxation_factor=0.01)
        flow_slow = FlowField(im=5, jm=20, km=9)
        flow_slow.initialize()  # 創建 3D 陣列
        flow_slow.p[:] = 101325.0
        nrbc_slow.apply_exit_nrbc(flow_slow, j_index=-1, target_p=target_p)
        delta_slow = abs(flow_slow.p[0, -1, 0] - 101325.0)

        # 較大的鬆弛因子
        nrbc_fast = NonReflectingBoundary(gas, relaxation_factor=0.5)
        flow_fast = FlowField(im=5, jm=20, km=9)
        flow_fast.initialize()  # 創建 3D 陣列
        flow_fast.p[:] = 101325.0
        nrbc_fast.apply_exit_nrbc(flow_fast, j_index=-1, target_p=target_p)
        delta_fast = abs(flow_fast.p[0, -1, 0] - 101325.0)

        # 較大的鬆弛因子應該導致更大的變化
        assert delta_fast > delta_slow


class TestModuleImports:
    """模組導入測試。"""

    def test_import_mixing_plane_model(self) -> None:
        """測試導入 MixingPlaneModel。"""
        from multall_turbomachinery_design.multall import MixingPlaneModel

        assert MixingPlaneModel is not None

    def test_import_mixing_plane_interface(self) -> None:
        """測試導入 MixingPlaneInterface。"""
        from multall_turbomachinery_design.multall import MixingPlaneInterface

        assert MixingPlaneInterface is not None

    def test_import_mixing_plane_type(self) -> None:
        """測試導入 MixingPlaneType。"""
        from multall_turbomachinery_design.multall import MixingPlaneType

        assert MixingPlaneType is not None

    def test_import_non_reflecting_boundary(self) -> None:
        """測試導入 NonReflectingBoundary。"""
        from multall_turbomachinery_design.multall import NonReflectingBoundary

        assert NonReflectingBoundary is not None
