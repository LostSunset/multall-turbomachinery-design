# -*- coding: utf-8 -*-
"""氣體性質計算模組。

提供氣體和蒸汽的熱力性質計算功能，對應 FORTRAN PROPS 子程序。
"""

from __future__ import annotations

import math

from .data_structures import FlowState, GasProperties


class PerfectGasCalculator:
    """完美氣體性質計算器。"""

    def __init__(self, gas: GasProperties) -> None:
        """初始化完美氣體計算器。

        Args:
            gas: 氣體性質參數
        """
        self.rgas = gas.rgas
        self.gamma = gas.gamma
        self.cp = self.rgas * self.gamma / (self.gamma - 1.0)
        self.cv = self.cp / self.gamma

    def calculate_properties(
        self,
        ho: float,
        s: float,
        v: float,
        poin: float,
        hoin: float,
        sin: float,
    ) -> FlowState:
        """計算完美氣體的流動狀態。

        Args:
            ho: 總焓 [J/kg]
            s: 熵 [J/(kg·K)]
            v: 速度 [m/s]
            poin: 進口總壓 [Pa]
            hoin: 進口總焓 [J/kg]
            sin: 進口熵 [J/(kg·K)]

        Returns:
            流動狀態
        """
        state = FlowState(ho=ho, s=s, v=v)

        # 靜焓
        state.h = ho - 0.5 * v * v

        # 靜溫
        state.t = state.h / self.cp

        # 等熵壓力比指數
        fga = self.gamma / (self.gamma - 1.0)

        # 總壓（從進口條件）
        state.po = poin * math.pow(ho / hoin, fga) * math.exp((sin - s) / self.rgas)

        # 靜壓
        state.p = state.po * math.pow(state.h / ho, fga)

        # 密度
        state.rho = state.p / (self.rgas * state.t)

        # 聲速
        state.vs = math.sqrt(self.gamma * self.rgas * state.t)

        # 馬赫數
        state.mach = v / state.vs if state.vs > 0 else 0.0

        # 總溫
        state.to = ho / self.cp

        return state

    def calculate_total_conditions(self, p: float, t: float, v: float) -> tuple[float, float]:
        """從靜態條件計算總壓和總溫。

        Args:
            p: 靜壓 [Pa]
            t: 靜溫 [K]
            v: 速度 [m/s]

        Returns:
            (總壓 [Pa], 總溫 [K])
        """
        # 馬赫數
        vs = math.sqrt(self.gamma * self.rgas * t)
        mach = v / vs if vs > 0 else 0.0

        # 總壓和總溫比
        temp_ratio = 1.0 + 0.5 * (self.gamma - 1.0) * mach * mach
        po = p * math.pow(temp_ratio, self.gamma / (self.gamma - 1.0))
        to = t * temp_ratio

        return po, to

    def calculate_enthalpy_from_temperature(self, t: float) -> float:
        """從溫度計算焓。

        Args:
            t: 溫度 [K]

        Returns:
            焓 [J/kg]
        """
        return self.cp * t

    def calculate_entropy_change(self, p1: float, t1: float, p2: float, t2: float) -> float:
        """計算熵變。

        Args:
            p1: 初始壓力 [Pa]
            t1: 初始溫度 [K]
            p2: 最終壓力 [Pa]
            t2: 最終溫度 [K]

        Returns:
            熵變 [J/(kg·K)]
        """
        ds = self.cp * math.log(t2 / t1) - self.rgas * math.log(p2 / p1)
        return ds


class SteamPropertiesCalculator:
    """蒸汽性質計算器（簡化版）。

    注意：這是簡化實現，完整版本需要蒸汽表或更複雜的多項式擬合。
    """

    def __init__(self) -> None:
        """初始化蒸汽性質計算器。"""
        # 水的性質常數
        self.rgas_steam = 461.5  # J/(kg·K)
        self.gamma_steam = 1.3  # 近似值

        # 使用完美氣體近似作為基礎
        self.gas_calc = PerfectGasCalculator(
            GasProperties(
                rgas=self.rgas_steam,
                gamma=self.gamma_steam,
                poin=1.0,
                toin=373.15,
            )
        )

    def calculate_properties(
        self,
        ho: float,
        s: float,
        v: float,
        poin: float,
        hoin: float,
        sin: float,
    ) -> FlowState:
        """計算蒸汽的流動狀態。

        Args:
            ho: 總焓 [J/kg]
            s: 熵 [J/(kg·K)]
            v: 速度 [m/s]
            poin: 進口總壓 [Pa]
            hoin: 進口總焓 [J/kg]
            sin: 進口熵 [J/(kg·K)]

        Returns:
            流動狀態
        """
        # 使用簡化的完美氣體近似
        # 實際應用中應使用 IAPWS-IF97 標準或查表法
        state = self.gas_calc.calculate_properties(ho, s, v, poin, hoin, sin)

        # TODO: 實現完整的蒸汽性質計算
        # - 判斷是否在濕蒸汽區
        # - 計算濕度分數
        # - 使用蒸汽表修正

        return state


def create_properties_calculator(
    gas: GasProperties, use_steam: bool = False
) -> PerfectGasCalculator | SteamPropertiesCalculator:
    """創建性質計算器工廠函數。

    Args:
        gas: 氣體性質
        use_steam: 是否使用蒸汽性質

    Returns:
        性質計算器
    """
    if use_steam:
        return SteamPropertiesCalculator()
    else:
        return PerfectGasCalculator(gas)
