# -*- coding: utf-8 -*-
"""MEANGEN 基本功能測試。"""

from __future__ import annotations

import math

import pytest

from multall_turbomachinery_design.meangen.data_structures import (
    CONSTANTS,
    FlowType,
    GasProperties,
    InputType,
    MachineType,
)
from multall_turbomachinery_design.meangen.gas_properties import PerfectGasCalculator
from multall_turbomachinery_design.meangen.velocity_triangles import (
    VelocityTriangleCalculator,
)


class TestDataStructures:
    """測試數據結構。"""

    def test_machine_type_enum(self) -> None:
        """測試機械類型枚舉。"""
        assert MachineType.TURBINE.value == "T"
        assert MachineType.COMPRESSOR.value == "C"

    def test_flow_type_enum(self) -> None:
        """測試流動類型枚舉。"""
        assert FlowType.AXIAL.value == "AXI"
        assert FlowType.MIXED.value == "MIX"

    def test_input_type_enum(self) -> None:
        """測試輸入類型枚舉。"""
        assert InputType.TYPE_A.value == "A"
        assert InputType.TYPE_B.value == "B"

    def test_constants(self) -> None:
        """測試常數定義。"""
        assert abs(CONSTANTS["DEG2RAD"] - math.pi / 180) < 1e-10
        assert abs(CONSTANTS["RAD2DEG"] - 180 / math.pi) < 1e-10
        assert abs(CONSTANTS["PI"] - math.pi) < 1e-10

    def test_gas_properties(self) -> None:
        """測試氣體性質數據結構。"""
        gas = GasProperties(rgas=287.5, gamma=1.4, poin=1.0, toin=300.0)
        assert gas.rgas == 287.5
        assert gas.gamma == 1.4
        assert gas.poin == 1.0
        assert gas.toin == 300.0


class TestPerfectGasCalculator:
    """測試完美氣體計算器。"""

    @pytest.fixture
    def air_properties(self) -> GasProperties:
        """空氣性質。"""
        return GasProperties(rgas=287.5, gamma=1.4, poin=1.0e5, toin=300.0)

    @pytest.fixture
    def gas_calc(self, air_properties: GasProperties) -> PerfectGasCalculator:
        """創建氣體計算器。"""
        return PerfectGasCalculator(air_properties)

    def test_initialization(self, gas_calc: PerfectGasCalculator) -> None:
        """測試初始化。"""
        assert gas_calc.rgas == 287.5
        assert gas_calc.gamma == 1.4
        # Cp = R·γ/(γ-1)
        expected_cp = 287.5 * 1.4 / 0.4
        assert abs(gas_calc.cp - expected_cp) < 1e-6

    def test_enthalpy_from_temperature(self, gas_calc: PerfectGasCalculator) -> None:
        """測試從溫度計算焓。"""
        t = 300.0  # K
        h = gas_calc.calculate_enthalpy_from_temperature(t)
        expected_h = gas_calc.cp * t
        assert abs(h - expected_h) < 1e-6

    def test_total_conditions(self, gas_calc: PerfectGasCalculator) -> None:
        """測試總壓總溫計算。"""
        p = 1.0e5  # Pa
        t = 288.15  # K
        v = 100.0  # m/s

        po, to = gas_calc.calculate_total_conditions(p, t, v)

        # 檢查總溫增加
        assert to > t
        # 檢查總壓增加
        assert po > p

        # 驗證等熵關係
        vs = math.sqrt(gas_calc.gamma * gas_calc.rgas * t)
        mach = v / vs
        temp_ratio = 1.0 + 0.5 * (gas_calc.gamma - 1.0) * mach * mach
        expected_to = t * temp_ratio
        assert abs(to - expected_to) < 1e-3


class TestVelocityTriangleCalculator:
    """測試速度三角形計算器。"""

    def test_turbine_type_a(self) -> None:
        """測試渦輪 Type A 速度三角形計算。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        phi = 0.6  # 流量係數
        psi = 2.0  # 負荷係數
        reaction = 0.5  # 反應度
        u = 200.0  # 圓周速度 [m/s]

        alpha_in, alpha_out, beta_in, beta_out = calc.calculate_type_a(phi, psi, reaction, u)

        # 檢查角度在合理範圍內
        assert -90 <= alpha_in <= 90
        assert -90 <= alpha_out <= 90
        assert -90 <= beta_in <= 90
        assert -90 <= beta_out <= 90

        # 對於渦輪，進出口絕對角應該相同（軸向重複級）
        assert abs(alpha_in - alpha_out) < 1.0

    def test_compressor_type_a(self) -> None:
        """測試壓縮機 Type A 速度三角形計算。"""
        calc = VelocityTriangleCalculator(MachineType.COMPRESSOR)

        phi = 0.5  # 流量係數
        psi = 0.4  # 負荷係數
        reaction = 0.5  # 反應度
        u = 200.0  # 圓周速度 [m/s]

        alpha_in, alpha_out, beta_in, beta_out = calc.calculate_type_a(phi, psi, reaction, u)

        # 檢查角度在合理範圍內
        assert -90 <= alpha_in <= 90
        assert -90 <= alpha_out <= 90
        assert -90 <= beta_in <= 90
        assert -90 <= beta_out <= 90

        # 壓縮機進口通常為軸向
        assert abs(alpha_in) < 5.0

    def test_flow_coefficient_calculation(self) -> None:
        """測試流量係數計算。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        vm = 100.0  # m/s
        u = 200.0  # m/s

        phi = calc.calculate_flow_coefficient(vm, u)
        assert abs(phi - 0.5) < 1e-6

    def test_loading_coefficient_calculation(self) -> None:
        """測試負荷係數計算。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        dh = 80000.0  # J/kg
        u = 200.0  # m/s

        psi = calc.calculate_loading_coefficient(dh, u)
        assert abs(psi - 2.0) < 1e-6

    def test_reaction_calculation(self) -> None:
        """測試反應度計算。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        dh_rotor = 40000.0  # J/kg
        dh_stage = 80000.0  # J/kg

        reaction = calc.calculate_reaction(dh_rotor, dh_stage)
        assert abs(reaction - 0.5) < 1e-6

    def test_create_velocity_triangle(self) -> None:
        """測試創建速度三角形。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        u = 200.0  # m/s
        vm = 100.0  # m/s
        alpha = 30.0  # 度

        vt = calc.create_velocity_triangle(u, vm, alpha, is_rotor=True)

        # 檢查基本屬性
        assert vt.vm == vm
        assert vt.u == u
        assert vt.alpha == alpha

        # 檢查切向速度
        expected_vtheta = vm * math.tan(alpha * CONSTANTS["DEG2RAD"])
        assert abs(vt.vtheta - expected_vtheta) < 1e-3

    def test_free_vortex_application(self) -> None:
        """測試自由渦應用。"""
        calc = VelocityTriangleCalculator(MachineType.TURBINE)

        # 設計點速度三角形
        u_design = 200.0
        vm_design = 100.0
        alpha_design = 30.0

        vt_design = calc.create_velocity_triangle(u_design, vm_design, alpha_design)

        # 應用自由渦到不同半徑
        r_design = 0.5  # m
        r_local = 0.6  # m
        frac_twist = 1.0  # 完全自由渦

        vt_local = calc.apply_free_vortex(vt_design, r_design, r_local, frac_twist)

        # 檢查 r·Vθ 守恆
        swirl_design = r_design * vt_design.vtheta
        swirl_local = r_local * vt_local.vtheta
        assert abs(swirl_design - swirl_local) < 1e-3


class TestUTF8Support:
    """測試 UTF-8 / 正體中文支援。"""

    def test_chinese_docstrings(self) -> None:
        """測試中文文檔字串。"""
        from multall_turbomachinery_design.meangen.data_structures import MachineType

        # 確保可以訪問帶中文註釋的模組
        assert MachineType.TURBINE is not None
        assert MachineType.COMPRESSOR is not None

    def test_chinese_variable_in_test(self) -> None:
        """測試中文變數處理。"""
        test_string = "渦輪機械"  # noqa: N806
        assert len(test_string) == 4
        assert test_string == "渦輪機械"
