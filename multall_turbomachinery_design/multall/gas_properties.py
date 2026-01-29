# -*- coding: utf-8 -*-
"""MULTALL 氣體性質計算。

提供完美氣體和變 CP 氣體的熱力學計算。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .data_structures import GasProperties, GasType


class GasCalculator:
    """氣體性質計算器。"""

    def __init__(self, gas: GasProperties) -> None:
        """初始化計算器。

        Args:
            gas: 氣體性質
        """
        self.gas = gas
        self._update_derived_properties()

    def _update_derived_properties(self) -> None:
        """更新派生性質。"""
        g = self.gas
        self.cp = g.cp
        self.gamma = g.gamma
        self.rgas = g.rgas
        self.cv = g.cv
        self.prandtl = g.prandtl

        # 預計算常用值
        self.ga1 = g.gamma - 1.0
        self.fga = (g.gamma + 1.0) / (2.0 * g.gamma)
        self.rcp = 1.0 / g.cp
        self.rcv = 1.0 / g.cv
        self.ga_ratio = g.gamma / g.ga1

    def enthalpy_from_temperature(
        self, t: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """從溫度計算焓。

        Args:
            t: 靜溫 [K]

        Returns:
            焓 [J/kg]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return self.cp * t
        else:
            # 變 CP: h = CP1*T + CP2*T^2/2 + CP3*T^3/3
            g = self.gas
            return g.cp1 * t + 0.5 * g.cp2 * t**2 + (1.0 / 3.0) * g.cp3 * t**3

    def temperature_from_enthalpy(self, h: float) -> float:
        """從焓計算溫度。

        Args:
            h: 焓 [J/kg]

        Returns:
            靜溫 [K]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return h / self.cp
        else:
            # 牛頓迭代求解
            g = self.gas
            t = h / g.cp1  # 初始猜測
            for _ in range(20):
                h_calc = self.enthalpy_from_temperature(t)
                cp_t = g.cp1 + g.cp2 * t + g.cp3 * t**2
                t_new = t - (h_calc - h) / cp_t
                if abs(t_new - t) < 1e-6:
                    break
                t = t_new
            return t

    def cp_at_temperature(self, t: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """計算指定溫度的 CP。

        Args:
            t: 溫度 [K]

        Returns:
            CP [J/(kg·K)]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return self.cp
        else:
            g = self.gas
            return g.cp1 + g.cp2 * t + g.cp3 * t**2

    def gamma_at_temperature(self, t: float) -> float:
        """計算指定溫度的比熱比。

        Args:
            t: 溫度 [K]

        Returns:
            比熱比 γ
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return self.gamma
        else:
            cp = self.cp_at_temperature(t)
            cv = cp - self.rgas
            return cp / cv

    def speed_of_sound(self, t: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """計算聲速。

        Args:
            t: 靜溫 [K]

        Returns:
            聲速 [m/s]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return np.sqrt(self.gamma * self.rgas * t)
        else:
            gamma = self.gamma_at_temperature(float(np.mean(t)))
            return np.sqrt(gamma * self.rgas * t)

    def total_temperature(
        self, t_static: float | NDArray[np.float64], v: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """計算總溫。

        Args:
            t_static: 靜溫 [K]
            v: 速度 [m/s]

        Returns:
            總溫 [K]
        """
        h_static = self.enthalpy_from_temperature(t_static)
        h_total = h_static + 0.5 * v**2
        return self.temperature_from_enthalpy(float(h_total))

    def total_pressure(
        self,
        p_static: float | NDArray[np.float64],
        t_static: float | NDArray[np.float64],
        t_total: float | NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """計算總壓。

        Args:
            p_static: 靜壓 [Pa]
            t_static: 靜溫 [K]
            t_total: 總溫 [K]

        Returns:
            總壓 [Pa]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            return p_static * (t_total / t_static) ** self.ga_ratio
        else:
            # 變 CP 需要積分
            gamma_avg = self.gamma_at_temperature(float(np.mean([t_static, t_total])))
            ga_ratio = gamma_avg / (gamma_avg - 1.0)
            return p_static * (t_total / t_static) ** ga_ratio

    def static_from_total(
        self,
        p_total: float,
        t_total: float,
        mach: float,
    ) -> tuple[float, float]:
        """從總條件和馬赫數計算靜態條件。

        Args:
            p_total: 總壓 [Pa]
            t_total: 總溫 [K]
            mach: 馬赫數

        Returns:
            (靜壓, 靜溫) [Pa, K]
        """
        if self.gas.gas_type == GasType.PERFECT_GAS:
            # 等熵關係
            t_ratio = 1.0 + 0.5 * self.ga1 * mach**2
            t_static = t_total / t_ratio
            p_static = p_total / (t_ratio**self.ga_ratio)
        else:
            # 迭代求解
            t_static = t_total / (1.0 + 0.5 * self.ga1 * mach**2)
            for _ in range(10):
                gamma = self.gamma_at_temperature(t_static)
                ga1 = gamma - 1.0
                t_ratio = 1.0 + 0.5 * ga1 * mach**2
                t_static_new = t_total / t_ratio
                if abs(t_static_new - t_static) < 0.01:
                    break
                t_static = t_static_new

            ga_ratio = gamma / ga1
            p_static = p_total / (t_ratio**ga_ratio)

        return p_static, t_static

    def mach_from_velocity(
        self, v: float | NDArray[np.float64], t_static: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """從速度和靜溫計算馬赫數。

        Args:
            v: 速度 [m/s]
            t_static: 靜溫 [K]

        Returns:
            馬赫數
        """
        a = self.speed_of_sound(t_static)
        return v / a

    def density(
        self, p: float | NDArray[np.float64], t: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """計算密度。

        Args:
            p: 壓力 [Pa]
            t: 溫度 [K]

        Returns:
            密度 [kg/m³]
        """
        return p / (self.rgas * t)

    def pressure(
        self, rho: float | NDArray[np.float64], t: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        """計算壓力。

        Args:
            rho: 密度 [kg/m³]
            t: 溫度 [K]

        Returns:
            壓力 [Pa]
        """
        return rho * self.rgas * t

    def internal_energy(self, t: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """計算內能。

        Args:
            t: 溫度 [K]

        Returns:
            內能 [J/kg]
        """
        return self.cv * t

    def total_enthalpy(
        self,
        t_static: float | NDArray[np.float64],
        vx: float | NDArray[np.float64],
        vr: float | NDArray[np.float64],
        vt: float | NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """計算總焓。

        Args:
            t_static: 靜溫 [K]
            vx: 軸向速度 [m/s]
            vr: 徑向速度 [m/s]
            vt: 切向速度 [m/s]

        Returns:
            總焓 [J/kg]
        """
        h_static = self.enthalpy_from_temperature(t_static)
        v_sq = vx**2 + vr**2 + vt**2
        return h_static + 0.5 * v_sq

    def rothalpy(
        self,
        ho: float | NDArray[np.float64],
        omega: float,
        r: float | NDArray[np.float64],
        vt: float | NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """計算旋轉焓（轉子中守恆）。

        I = H0 - ω*r*Vθ

        Args:
            ho: 總焓 [J/kg]
            omega: 角速度 [rad/s]
            r: 半徑 [m]
            vt: 切向速度 [m/s]

        Returns:
            旋轉焓 [J/kg]
        """
        return ho - omega * r * vt

    def critical_velocity(self, t_total: float) -> float:
        """計算臨界速度（M=1 時的速度）。

        Args:
            t_total: 總溫 [K]

        Returns:
            臨界速度 [m/s]
        """
        # V* = sqrt(2*γ/(γ+1) * R * T0)
        h_total = self.enthalpy_from_temperature(t_total)
        return math.sqrt(2.0 * self.fga * h_total)

    def isentropic_relations(self, mach: float) -> tuple[float, float, float, float]:
        """計算等熵關係。

        Args:
            mach: 馬赫數

        Returns:
            (T/T0, P/P0, ρ/ρ0, A/A*)
        """
        ga1 = self.ga1
        gamma = self.gamma

        mach_sq = mach**2
        t_ratio = 1.0 / (1.0 + 0.5 * ga1 * mach_sq)
        p_ratio = t_ratio ** (gamma / ga1)
        rho_ratio = t_ratio ** (1.0 / ga1)

        # 面積比
        if mach > 0.001:
            term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * ga1 * mach_sq)
            a_ratio = (1.0 / mach) * term ** ((gamma + 1.0) / (2.0 * ga1))
        else:
            a_ratio = 1e10  # 接近無窮大

        return t_ratio, p_ratio, rho_ratio, a_ratio


def create_air_calculator() -> GasCalculator:
    """創建空氣計算器（預設完美氣體）。

    Returns:
        氣體計算器
    """
    gas = GasProperties(
        cp=1005.0,
        gamma=1.4,
        gas_type=GasType.PERFECT_GAS,
    )
    return GasCalculator(gas)


def create_combustion_gas_calculator(
    cp1: float = 1272.5,
    cp2: float = 0.2125,
    cp3: float = 0.000015625,
    tref: float = 1400.0,
    rgas: float = 287.15,
) -> GasCalculator:
    """創建燃氣計算器（變 CP）。

    Args:
        cp1: CP 常數項
        cp2: CP 線性項
        cp3: CP 二次項
        tref: 參考溫度
        rgas: 氣體常數

    Returns:
        氣體計算器
    """
    cp_ref = cp1 + cp2 * tref + cp3 * tref**2
    cv_ref = cp_ref - rgas
    gamma_ref = cp_ref / cv_ref

    gas = GasProperties(
        cp=cp_ref,
        gamma=gamma_ref,
        rgas=rgas,
        gas_type=GasType.VARIABLE_CP,
        cp1=cp1,
        cp2=cp2,
        cp3=cp3,
        tref=tref,
    )
    return GasCalculator(gas)
