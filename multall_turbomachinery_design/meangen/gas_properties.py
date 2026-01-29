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
    """蒸汽性質計算器（IAPWS-IF97）。

    實現工業用水和蒸汽性質計算，基於 IAPWS-IF97 標準。
    支援過熱蒸汽和濕蒸汽區計算。

    當 iapws 庫可用時使用完整實現，否則使用近似計算。
    """

    # 水的臨界點性質
    T_CRIT = 647.096  # K
    P_CRIT = 22.064e6  # Pa
    RHO_CRIT = 322.0  # kg/m³

    # 氣體常數
    R_STEAM = 461.526  # J/(kg·K)

    def __init__(self) -> None:
        """初始化蒸汽性質計算器。"""
        # 檢查 iapws 庫是否可用
        self._iapws_available = False
        try:
            import iapws

            self._iapws = iapws
            self._iapws_available = True
        except ImportError:
            self._iapws = None

        # 回退用的完美氣體計算器
        self._fallback_calc = PerfectGasCalculator(
            GasProperties(
                rgas=self.R_STEAM,
                gamma=1.3,
                poin=1.0,
                toin=373.15,
            )
        )

    @property
    def is_iapws_available(self) -> bool:
        """檢查 IAPWS 庫是否可用。"""
        return self._iapws_available

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
        state = FlowState(ho=ho, s=s, v=v)

        # 靜焓
        h = ho - 0.5 * v * v
        state.h = h

        if self._iapws_available:
            # 使用 IAPWS-IF97 計算
            state = self._calculate_with_iapws(state, ho, h, s, v, poin, hoin, sin)
        else:
            # 使用近似計算
            state = self._calculate_approximate(state, ho, h, s, v, poin, hoin, sin)

        return state

    def _calculate_with_iapws(
        self,
        state: FlowState,
        ho: float,
        h: float,
        s: float,
        v: float,
        poin: float,
        hoin: float,
        sin: float,
    ) -> FlowState:
        """使用 IAPWS-IF97 計算蒸汽性質。"""
        from iapws import IAPWS97

        # 轉換單位：J/kg -> kJ/kg, J/(kg·K) -> kJ/(kg·K)
        h_kj = h / 1000.0
        s_kj = s / 1000.0

        try:
            # 嘗試用焓和熵確定狀態
            steam = IAPWS97(h=h_kj, s=s_kj)

            if steam.status:
                # IAPWS 計算成功
                state.t = steam.T  # K
                state.p = steam.P * 1e6  # MPa -> Pa
                state.rho = steam.rho  # kg/m³

                # 計算聲速
                # c = sqrt(dp/drho)_s ≈ sqrt(gamma * p / rho)
                # 對於蒸汽，使用更精確的計算
                gamma_eff = steam.cp / steam.cv if steam.cv > 0 else 1.3
                state.vs = math.sqrt(gamma_eff * state.p / state.rho)

                # 馬赫數
                state.mach = v / state.vs if state.vs > 0 else 0.0

                # 總條件
                ho_kj = ho / 1000.0
                steam_total = IAPWS97(h=ho_kj, s=s_kj)
                if steam_total.status:
                    state.to = steam_total.T
                    state.po = steam_total.P * 1e6
                else:
                    state.to = state.t + v * v / (2 * steam.cp * 1000)
                    state.po = state.p * (state.to / state.t) ** (gamma_eff / (gamma_eff - 1))

                # 濕度（如果在濕蒸汽區）
                if hasattr(steam, "x") and steam.x is not None:
                    state.wetness = 1.0 - steam.x  # x 是乾度
                else:
                    state.wetness = 0.0

                return state

        except Exception:
            pass

        # 如果 IAPWS 計算失敗，使用近似方法
        return self._calculate_approximate(state, ho, h, s, v, poin, hoin, sin)

    def _calculate_approximate(
        self,
        state: FlowState,
        ho: float,
        h: float,
        s: float,
        v: float,
        poin: float,
        hoin: float,
        sin: float,
    ) -> FlowState:
        """使用近似方法計算蒸汽性質。"""
        # 近似比熱（過熱蒸汽）
        cp_approx = 2000.0  # J/(kg·K)
        gamma_approx = 1.3

        # 估算溫度
        state.t = h / cp_approx
        if state.t < 273.15:
            state.t = 273.15 + h / cp_approx  # 修正負溫度

        # 等熵壓力比指數
        fga = gamma_approx / (gamma_approx - 1.0)

        # 總壓（從進口條件）
        if hoin > 0:
            state.po = poin * math.pow(ho / hoin, fga) * math.exp((sin - s) / self.R_STEAM)
        else:
            state.po = poin

        # 靜壓
        if ho > 0:
            state.p = state.po * math.pow(h / ho, fga)
        else:
            state.p = state.po

        # 密度
        state.rho = state.p / (self.R_STEAM * state.t)

        # 聲速
        state.vs = math.sqrt(gamma_approx * self.R_STEAM * state.t)

        # 馬赫數
        state.mach = v / state.vs if state.vs > 0 else 0.0

        # 總溫
        state.to = ho / cp_approx

        # 濕度估算
        state.wetness = self._estimate_wetness(state.p, state.t, h)

        return state

    def _estimate_wetness(self, p: float, t: float, h: float) -> float:
        """估算濕度。

        Args:
            p: 壓力 [Pa]
            t: 溫度 [K]
            h: 焓 [J/kg]

        Returns:
            濕度分數 (0=過熱, 1=飽和水)
        """
        # 飽和溫度近似（簡化）
        t_sat = self.saturation_temperature(p)

        if t > t_sat + 5.0:
            # 過熱蒸汽
            return 0.0

        # 飽和性質近似
        h_f, h_g = self.saturation_enthalpies(p)

        if h >= h_g:
            # 過熱
            return 0.0
        elif h <= h_f:
            # 過冷液體
            return 1.0
        else:
            # 濕蒸汽區
            x = (h - h_f) / (h_g - h_f)  # 乾度
            return 1.0 - x

    def saturation_temperature(self, p: float) -> float:
        """計算飽和溫度。

        使用 Antoine 方程近似。

        Args:
            p: 壓力 [Pa]

        Returns:
            飽和溫度 [K]
        """
        # Antoine 常數（水，使用 Pa）
        # log10(P) = A - B / (C + T)
        # 這裡使用簡化的逆向計算
        p_bar = p / 1e5

        if p_bar <= 0:
            return 373.15

        # 簡化的飽和線近似
        # T_sat ≈ 373.15 + 42 * (log10(p_bar))
        try:
            t_sat = 373.15 + 42.0 * math.log10(max(p_bar, 0.01))
        except ValueError:
            t_sat = 373.15

        # 限制範圍
        return max(273.15, min(t_sat, self.T_CRIT))

    def saturation_enthalpies(self, p: float) -> tuple[float, float]:
        """計算飽和焓。

        Args:
            p: 壓力 [Pa]

        Returns:
            (飽和液焓 [J/kg], 飽和氣焓 [J/kg])
        """
        if self._iapws_available:
            try:
                from iapws import IAPWS97

                steam_g = IAPWS97(P=p / 1e6, x=1)  # 飽和氣
                steam_f = IAPWS97(P=p / 1e6, x=0)  # 飽和液

                if steam_g.status and steam_f.status:
                    return steam_f.h * 1000, steam_g.h * 1000
            except Exception:
                pass

        # 近似計算
        t_sat = self.saturation_temperature(p)

        # 近似飽和焓
        # h_f ≈ 4.18 * (T - 273.15) kJ/kg
        h_f = 4180 * (t_sat - 273.15)

        # h_fg 隨壓力變化，使用近似
        # h_fg ≈ 2500 - 2.4 * (T - 273.15) kJ/kg
        h_fg = (2500 - 2.4 * (t_sat - 273.15)) * 1000

        h_g = h_f + h_fg

        return h_f, h_g

    def saturation_pressure(self, t: float) -> float:
        """計算飽和壓力。

        Args:
            t: 溫度 [K]

        Returns:
            飽和壓力 [Pa]
        """
        if self._iapws_available:
            try:
                from iapws import IAPWS97

                steam = IAPWS97(T=t, x=0)
                if steam.status:
                    return steam.P * 1e6
            except Exception:
                pass

        # Antoine 方程近似
        # log10(P_bar) ≈ (T - 373.15) / 42
        try:
            log_p = (t - 373.15) / 42.0
            p_bar = 10**log_p
        except OverflowError:
            p_bar = 1.0

        return p_bar * 1e5

    def calculate_from_pt(self, p: float, t: float) -> dict[str, float]:
        """從壓力和溫度計算蒸汽性質。

        Args:
            p: 壓力 [Pa]
            t: 溫度 [K]

        Returns:
            包含 h, s, rho, cp, cv, vs 的字典
        """
        if self._iapws_available:
            try:
                from iapws import IAPWS97

                steam = IAPWS97(P=p / 1e6, T=t)
                if steam.status:
                    gamma = steam.cp / steam.cv if steam.cv > 0 else 1.3
                    return {
                        "h": steam.h * 1000,  # J/kg
                        "s": steam.s * 1000,  # J/(kg·K)
                        "rho": steam.rho,
                        "cp": steam.cp * 1000,
                        "cv": steam.cv * 1000,
                        "vs": math.sqrt(gamma * p / steam.rho),
                    }
            except Exception:
                pass

        # 近似計算
        cp = 2000.0
        cv = cp / 1.3
        rho = p / (self.R_STEAM * t)
        h = cp * t
        s = cp * math.log(t / 273.15) - self.R_STEAM * math.log(p / 101325)
        vs = math.sqrt(1.3 * self.R_STEAM * t)

        return {"h": h, "s": s, "rho": rho, "cp": cp, "cv": cv, "vs": vs}


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
