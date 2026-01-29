# -*- coding: utf-8 -*-
"""MULTALL 黏性模型模組。

實現混合長度和 Spalart-Allmaras 紊流模型。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties, Grid3D, ViscousModel


class WallDistanceCalculator:
    """壁面距離計算器。

    計算每個網格點到最近壁面的距離。
    """

    def __init__(self, grid: Grid3D) -> None:
        """初始化壁面距離計算器。

        Args:
            grid: 網格數據
        """
        self.grid = grid
        self._wall_distance: NDArray[np.float64] | None = None

    def compute(self) -> NDArray[np.float64]:
        """計算壁面距離。

        對於渦輪機械，壁面通常是 HUB (K=0) 和 CASING (K=KM-1)。

        Returns:
            壁面距離數組
        """
        if self._wall_distance is not None:
            return self._wall_distance

        im, jm, km = self.grid.im, self.grid.jm, self.grid.km

        # 初始化壁面距離
        d = np.zeros((im, jm, km))

        # 假設 K 方向是跨向（從 HUB 到 CASING）
        # 壁面在 K=0 (HUB) 和 K=KM-1 (CASING)
        # 這裡使用簡化的線性插值

        for k in range(km):
            # 到 HUB 的距離比例
            frac_hub = k / max(km - 1, 1)
            # 到 CASING 的距離比例
            frac_cas = 1.0 - frac_hub

            # 使用較小的距離
            # 假設 span 是網格 K 方向的總高度
            span = 0.05  # 預設 5cm，實際應從網格獲取

            d_hub = frac_hub * span
            d_cas = frac_cas * span
            d[:, :, k] = np.minimum(d_hub, d_cas)

        self._wall_distance = d
        return d


class MixingLengthModel:
    """混合長度紊流模型。

    使用 Prandtl 混合長度假設計算渦黏性。
    μ_t = ρ * l_m^2 * |∂u/∂y|

    其中 l_m = κ * y * (1 - exp(-y+/A+))
    """

    # 模型常數
    KAPPA = 0.41  # von Karman 常數
    A_PLUS = 26.0  # van Driest 常數

    def __init__(self, gas: GasProperties) -> None:
        """初始化混合長度模型。

        Args:
            gas: 氣體性質
        """
        self.gas = gas
        self.wall_distance_calc: WallDistanceCalculator | None = None

    def compute_mixing_length(
        self,
        wall_distance: NDArray[np.float64],
        y_plus: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """計算混合長度。

        l_m = κ * y * D
        D = 1 - exp(-y+/A+)  # van Driest 阻尼函數

        Args:
            wall_distance: 壁面距離
            y_plus: 無因次壁面距離

        Returns:
            混合長度
        """
        # van Driest 阻尼函數
        damping = 1.0 - np.exp(-y_plus / self.A_PLUS)

        # 混合長度
        l_m = self.KAPPA * wall_distance * damping

        return l_m

    def compute_eddy_viscosity(
        self,
        flow: FlowField,
        grid: Grid3D,
        wall_distance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """計算渦黏性。

        μ_t = ρ * l_m^2 * |S|
        S = √(2 * S_ij * S_ij)  # 應變率張量的大小

        Args:
            flow: 流場數據
            grid: 網格數據
            wall_distance: 壁面距離

        Returns:
            渦黏性
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 計算應變率張量（簡化版本，只計算主要分量）
        # S = |∂u/∂y| 近似

        # 估計網格間距
        dy = 0.001  # 需要從 grid 獲取

        # 計算速度梯度 ∂Vx/∂K（跨向）
        dvx_dk = np.zeros((im, jm, km))
        dvr_dk = np.zeros((im, jm, km))
        dvt_dk = np.zeros((im, jm, km))

        for k in range(1, km - 1):
            dvx_dk[:, :, k] = (flow.vx[:, :, k + 1] - flow.vx[:, :, k - 1]) / (2 * dy)
            dvr_dk[:, :, k] = (flow.vr[:, :, k + 1] - flow.vr[:, :, k - 1]) / (2 * dy)
            dvt_dk[:, :, k] = (flow.vt[:, :, k + 1] - flow.vt[:, :, k - 1]) / (2 * dy)

        # 邊界
        dvx_dk[:, :, 0] = (flow.vx[:, :, 1] - flow.vx[:, :, 0]) / dy
        dvx_dk[:, :, -1] = (flow.vx[:, :, -1] - flow.vx[:, :, -2]) / dy
        dvr_dk[:, :, 0] = (flow.vr[:, :, 1] - flow.vr[:, :, 0]) / dy
        dvr_dk[:, :, -1] = (flow.vr[:, :, -1] - flow.vr[:, :, -2]) / dy
        dvt_dk[:, :, 0] = (flow.vt[:, :, 1] - flow.vt[:, :, 0]) / dy
        dvt_dk[:, :, -1] = (flow.vt[:, :, -1] - flow.vt[:, :, -2]) / dy

        # 應變率大小
        strain_rate = np.sqrt(dvx_dk**2 + dvr_dk**2 + dvt_dk**2)

        # 計算壁面剪應力和 y+
        # τ_w = μ * (∂u/∂y)_wall
        mu = self._compute_dynamic_viscosity(flow.t_static)
        tau_w = np.abs(mu * strain_rate)
        u_tau = np.sqrt(np.maximum(tau_w, 1e-20) / np.maximum(flow.rho, 1e-10))

        # y+ = ρ * u_τ * y / μ
        y_plus = flow.rho * u_tau * wall_distance / np.maximum(mu, 1e-20)
        y_plus = np.maximum(y_plus, 0.1)  # 避免零

        # 混合長度
        l_m = self.compute_mixing_length(wall_distance, y_plus)

        # 渦黏性
        mu_t = flow.rho * l_m**2 * strain_rate

        return mu_t

    def _compute_dynamic_viscosity(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        """計算動力黏度（Sutherland 公式）。

        μ = μ_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

        Args:
            temperature: 溫度 [K]

        Returns:
            動力黏度 [Pa·s]
        """
        # Sutherland 常數（空氣）
        mu_ref = 1.716e-5  # Pa·s at T_ref
        t_ref = 273.15  # K
        s_const = 110.4  # K

        mu = mu_ref * (temperature / t_ref) ** 1.5 * (t_ref + s_const) / (temperature + s_const)

        return mu


class SpalartAllmarasModel:
    """Spalart-Allmaras 一方程紊流模型。

    求解修正運動黏度 ν̃ 的輸運方程。
    """

    # 模型常數
    CB1 = 0.1355
    CB2 = 0.622
    CV1 = 7.1
    CV2 = 0.7
    CV3 = 0.9
    CW1 = 3.2390678  # CB1/KAPPA^2 + (1+CB2)/SIGMA
    CW2 = 0.3
    CW3 = 2.0
    KAPPA = 0.41
    SIGMA = 2.0 / 3.0

    def __init__(self, gas: GasProperties) -> None:
        """初始化 Spalart-Allmaras 模型。

        Args:
            gas: 氣體性質
        """
        self.gas = gas

        # 修正運動黏度場
        self._nu_tilde: NDArray[np.float64] | None = None

    def initialize(self, flow: FlowField) -> None:
        """初始化修正運動黏度場。

        Args:
            flow: 流場數據
        """
        # 初始化為分子運動黏度的 3 倍
        nu = self._compute_kinematic_viscosity(flow)
        self._nu_tilde = 3.0 * nu

    def compute_eddy_viscosity(
        self,
        flow: FlowField,
        grid: Grid3D,
        wall_distance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """計算渦黏性。

        μ_t = ρ * ν̃ * f_v1

        Args:
            flow: 流場數據
            grid: 網格數據
            wall_distance: 壁面距離

        Returns:
            渦黏性
        """
        if self._nu_tilde is None:
            self.initialize(flow)

        nu = self._compute_kinematic_viscosity(flow)

        # χ = ν̃/ν
        chi = self._nu_tilde / np.maximum(nu, 1e-20)

        # f_v1 = χ^3 / (χ^3 + c_v1^3)
        chi3 = chi**3
        f_v1 = chi3 / (chi3 + self.CV1**3)

        # μ_t = ρ * ν̃ * f_v1
        mu_t = flow.rho * self._nu_tilde * f_v1

        return mu_t

    def compute_source_terms(
        self,
        flow: FlowField,
        grid: Grid3D,
        wall_distance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """計算 SA 方程的源項。

        P - D + (1/σ) * cb2 * (∇ν̃)^2

        Args:
            flow: 流場數據
            grid: 網格數據
            wall_distance: 壁面距離

        Returns:
            源項
        """
        if self._nu_tilde is None:
            self.initialize(flow)

        nu = self._compute_kinematic_viscosity(flow)

        # χ = ν̃/ν
        chi = self._nu_tilde / np.maximum(nu, 1e-20)

        # f_v1, f_v2
        chi3 = chi**3
        f_v1 = chi3 / (chi3 + self.CV1**3)
        f_v2 = 1.0 - chi / (1.0 + chi * f_v1)

        # 渦量（簡化）
        omega = self._compute_vorticity(flow, grid)

        # S̃ = Ω + ν̃/(κ^2 * d^2) * f_v2
        d2 = np.maximum(wall_distance**2, 1e-20)
        s_tilde = omega + self._nu_tilde / (self.KAPPA**2 * d2) * f_v2
        s_tilde = np.maximum(s_tilde, 0.3 * omega)

        # 產生項 P = cb1 * S̃ * ν̃
        production = self.CB1 * s_tilde * self._nu_tilde

        # r = ν̃ / (S̃ * κ^2 * d^2)
        r = self._nu_tilde / np.maximum(s_tilde * self.KAPPA**2 * d2, 1e-20)
        r = np.minimum(r, 10.0)

        # g = r + cw2 * (r^6 - r)
        g = r + self.CW2 * (r**6 - r)

        # f_w = g * ((1 + cw3^6) / (g^6 + cw3^6))^(1/6)
        f_w = g * ((1.0 + self.CW3**6) / (g**6 + self.CW3**6)) ** (1.0 / 6.0)

        # 耗散項 D = cw1 * f_w * (ν̃/d)^2
        destruction = self.CW1 * f_w * (self._nu_tilde / np.maximum(wall_distance, 1e-10)) ** 2

        # 源項
        source = production - destruction

        return source

    def update(
        self,
        flow: FlowField,
        grid: Grid3D,
        wall_distance: NDArray[np.float64],
        dt: NDArray[np.float64],
    ) -> None:
        """更新修正運動黏度場。

        Args:
            flow: 流場數據
            grid: 網格數據
            wall_distance: 壁面距離
            dt: 時間步長
        """
        if self._nu_tilde is None:
            self.initialize(flow)

        # 計算源項
        source = self.compute_source_terms(flow, grid, wall_distance)

        # 簡單顯式更新
        self._nu_tilde = self._nu_tilde + dt * source

        # 確保正值
        self._nu_tilde = np.maximum(self._nu_tilde, 1e-10)

        # 壁面邊界條件 ν̃ = 0
        self._nu_tilde[:, :, 0] = 0.0
        self._nu_tilde[:, :, -1] = 0.0

    def _compute_kinematic_viscosity(self, flow: FlowField) -> NDArray[np.float64]:
        """計算運動黏度。

        Args:
            flow: 流場數據

        Returns:
            運動黏度
        """
        # 動力黏度（Sutherland）
        mu_ref = 1.716e-5
        t_ref = 273.15
        s_const = 110.4

        mu = mu_ref * (flow.t_static / t_ref) ** 1.5 * (t_ref + s_const) / (flow.t_static + s_const)

        # 運動黏度 ν = μ/ρ
        nu = mu / np.maximum(flow.rho, 1e-10)

        return nu

    def _compute_vorticity(self, flow: FlowField, grid: Grid3D) -> NDArray[np.float64]:
        """計算渦量。

        Ω = |∇ × V|

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            渦量
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 簡化版本：只計算主要分量
        # 網格間距
        dr = 0.001

        omega = np.zeros((im, jm, km))

        # ∂Vt/∂r - ∂Vr/∂θ
        for k in range(1, km - 1):
            dvt_dr = (flow.vt[:, :, k + 1] - flow.vt[:, :, k - 1]) / (2 * dr)
            omega[:, :, k] = np.abs(dvt_dr)

        return omega


class ViscousFluxCalculator:
    """黏性通量計算器。

    計算黏性應力和熱傳導通量。
    """

    def __init__(
        self,
        gas: GasProperties,
        viscous_model: ViscousModel = ViscousModel.MIXING_LENGTH,
    ) -> None:
        """初始化黏性通量計算器。

        Args:
            gas: 氣體性質
            viscous_model: 黏性模型類型
        """
        self.gas = gas
        self.viscous_model = viscous_model

        # 根據模型類型創建對應的模型
        if viscous_model == ViscousModel.MIXING_LENGTH:
            self.turbulence_model = MixingLengthModel(gas)
        elif viscous_model == ViscousModel.SPALART_ALLMARAS:
            self.turbulence_model = SpalartAllmarasModel(gas)
        else:
            self.turbulence_model = None

        self.wall_distance_calc: WallDistanceCalculator | None = None

    def compute_viscous_flux(
        self,
        flow: FlowField,
        grid: Grid3D,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """計算黏性通量。

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            (X動量黏性通量, R動量黏性通量, θ動量黏性通量, 能量黏性通量, 渦黏性)
        """
        if self.viscous_model == ViscousModel.INVISCID:
            # 無黏性通量
            shape = (flow.im, flow.jm, flow.km)
            return (
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
            )

        # 計算壁面距離
        if self.wall_distance_calc is None:
            self.wall_distance_calc = WallDistanceCalculator(grid)
        wall_distance = self.wall_distance_calc.compute()

        # 計算分子黏度
        mu = self._compute_molecular_viscosity(flow.t_static)

        # 計算渦黏性
        if self.turbulence_model is not None:
            mu_t = self.turbulence_model.compute_eddy_viscosity(flow, grid, wall_distance)
        else:
            mu_t = np.zeros_like(mu)

        # 總黏度
        mu_total = mu + mu_t

        # 計算應變率張量
        strain_xx, strain_rr, strain_tt, strain_xr, strain_xt, strain_rt = (
            self._compute_strain_rate_tensor(flow, grid)
        )

        # 黏性應力
        # τ_xx = 2μ * S_xx - (2/3)μ * ∇·V
        div_v = strain_xx + strain_rr + strain_tt  # 近似

        tau_xx = 2.0 * mu_total * strain_xx - (2.0 / 3.0) * mu_total * div_v
        tau_rr = 2.0 * mu_total * strain_rr - (2.0 / 3.0) * mu_total * div_v
        tau_tt = 2.0 * mu_total * strain_tt - (2.0 / 3.0) * mu_total * div_v
        tau_xr = mu_total * strain_xr
        tau_xt = mu_total * strain_xt
        # tau_rt 用於完整 3D 應力張量（目前簡化）
        _tau_rt = mu_total * strain_rt

        # 熱傳導
        # q = -k * ∇T
        # k = μ * cp / Pr
        pr = self.gas.prandtl
        k = mu_total * self.gas.cp / pr

        dt_dx, dt_dr, _dt_dt = self._compute_temperature_gradient(flow, grid)
        q_x = -k * dt_dx
        # q_r 用於完整 3D 熱傳導（目前簡化）
        _q_r = -k * dt_dr

        # 黏性通量（簡化，只返回主要分量）
        # 能量方程中的黏性功和熱傳導
        viscous_work = tau_xx * flow.vx + tau_xr * flow.vr + tau_xt * flow.vt

        return tau_xx, tau_rr, tau_tt, viscous_work - q_x, mu_t

    def _compute_molecular_viscosity(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        """計算分子動力黏度。"""
        mu_ref = 1.716e-5
        t_ref = 273.15
        s_const = 110.4

        mu = mu_ref * (temperature / t_ref) ** 1.5 * (t_ref + s_const) / (temperature + s_const)

        return mu

    def _compute_strain_rate_tensor(
        self, flow: FlowField, grid: Grid3D
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """計算應變率張量。

        S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)

        Returns:
            (S_xx, S_rr, S_θθ, S_xr, S_xθ, S_rθ)
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 網格間距（簡化）
        dx = 0.001
        dr = 0.001
        dtheta = 0.001

        # 初始化
        s_xx = np.zeros((im, jm, km))
        s_rr = np.zeros((im, jm, km))
        s_tt = np.zeros((im, jm, km))
        s_xr = np.zeros((im, jm, km))
        s_xt = np.zeros((im, jm, km))
        s_rt = np.zeros((im, jm, km))

        # 計算速度梯度（中心差分）
        for j in range(1, jm - 1):
            s_xx[:, j, :] = (flow.vx[:, j + 1, :] - flow.vx[:, j - 1, :]) / (2 * dx)

        for k in range(1, km - 1):
            s_rr[:, :, k] = (flow.vr[:, :, k + 1] - flow.vr[:, :, k - 1]) / (2 * dr)

        for i in range(1, im - 1):
            s_tt[i, :, :] = (flow.vt[i + 1, :, :] - flow.vt[i - 1, :, :]) / (2 * dtheta)

        # 交叉項
        for j in range(1, jm - 1):
            for k in range(1, km - 1):
                dvx_dr = (flow.vx[:, j, k + 1] - flow.vx[:, j, k - 1]) / (2 * dr)
                dvr_dx = (flow.vr[:, j + 1, k] - flow.vr[:, j - 1, k]) / (2 * dx)
                s_xr[:, j, k] = 0.5 * (dvx_dr + dvr_dx)

        return s_xx, s_rr, s_tt, s_xr, s_xt, s_rt

    def _compute_temperature_gradient(
        self, flow: FlowField, grid: Grid3D
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """計算溫度梯度。

        Returns:
            (∂T/∂x, ∂T/∂r, ∂T/∂θ)
        """
        im, jm, km = flow.im, flow.jm, flow.km

        dx = 0.001
        dr = 0.001
        dtheta = 0.001

        dt_dx = np.zeros((im, jm, km))
        dt_dr = np.zeros((im, jm, km))
        dt_dt = np.zeros((im, jm, km))

        # X 方向
        for j in range(1, jm - 1):
            dt_dx[:, j, :] = (flow.t_static[:, j + 1, :] - flow.t_static[:, j - 1, :]) / (2 * dx)

        # R 方向
        for k in range(1, km - 1):
            dt_dr[:, :, k] = (flow.t_static[:, :, k + 1] - flow.t_static[:, :, k - 1]) / (2 * dr)

        # θ 方向
        for i in range(1, im - 1):
            dt_dt[i, :, :] = (flow.t_static[i + 1, :, :] - flow.t_static[i - 1, :, :]) / (
                2 * dtheta
            )

        return dt_dx, dt_dr, dt_dt
