# -*- coding: utf-8 -*-
"""MULTALL 通量計算模組。

實現對流通量和擴散通量的計算。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties, Grid3D
from .gas_properties import GasCalculator


class FluxCalculator:
    """通量計算器。

    計算有限體積法所需的對流通量和擴散通量。
    """

    def __init__(self, gas: GasProperties) -> None:
        """初始化通量計算器。

        Args:
            gas: 氣體性質
        """
        self.gas = gas
        self.gas_calc = GasCalculator(gas)

    def compute_convective_flux_x(
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
        """計算 X 方向（軸向）對流通量。

        使用 Roe 平均的通量分裂方法。

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            (質量通量, X動量通量, R動量通量, θ動量通量, 能量通量)
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 初始化通量數組（面上的通量）
        flux_mass = np.zeros((im, jm + 1, km))
        flux_momx = np.zeros((im, jm + 1, km))
        flux_momr = np.zeros((im, jm + 1, km))
        flux_momt = np.zeros((im, jm + 1, km))
        flux_energy = np.zeros((im, jm + 1, km))

        # 對每個內部面計算通量
        for j in range(1, jm):
            # 左右狀態
            rho_l = flow.rho[:, j - 1, :]
            rho_r = flow.rho[:, j, :]

            vx_l = flow.vx[:, j - 1, :]
            vx_r = flow.vx[:, j, :]

            vr_l = flow.vr[:, j - 1, :]
            vr_r = flow.vr[:, j, :]

            vt_l = flow.vt[:, j - 1, :]
            vt_r = flow.vt[:, j, :]

            p_l = flow.p[:, j - 1, :]
            p_r = flow.p[:, j, :]

            # 總焓
            ho_l = flow.ho[:, j - 1, :]
            ho_r = flow.ho[:, j, :]

            # Roe 平均
            sqrt_rho_l = np.sqrt(rho_l)
            sqrt_rho_r = np.sqrt(rho_r)
            denom = sqrt_rho_l + sqrt_rho_r

            rho_roe = sqrt_rho_l * sqrt_rho_r
            vx_roe = (sqrt_rho_l * vx_l + sqrt_rho_r * vx_r) / denom
            vr_roe = (sqrt_rho_l * vr_l + sqrt_rho_r * vr_r) / denom
            vt_roe = (sqrt_rho_l * vt_l + sqrt_rho_r * vt_r) / denom
            ho_roe = (sqrt_rho_l * ho_l + sqrt_rho_r * ho_r) / denom

            # 聲速
            v_sq = vx_roe**2 + vr_roe**2 + vt_roe**2
            a_sq = (self.gas.gamma - 1.0) * (ho_roe - 0.5 * v_sq)
            a_sq = np.maximum(a_sq, 1e-10)
            a_roe = np.sqrt(a_sq)

            # 特徵值
            lambda_1 = np.abs(vx_roe - a_roe)
            lambda_2 = np.abs(vx_roe)
            lambda_3 = np.abs(vx_roe + a_roe)

            # 熵修正（避免膨脹激波）
            eps = 0.1 * a_roe
            lambda_1 = np.where(lambda_1 < eps, (lambda_1**2 + eps**2) / (2 * eps), lambda_1)
            lambda_3 = np.where(lambda_3 < eps, (lambda_3**2 + eps**2) / (2 * eps), lambda_3)

            # 狀態差
            d_rho = rho_r - rho_l
            d_vx = vx_r - vx_l
            d_vr = vr_r - vr_l
            d_vt = vt_r - vt_l
            d_p = p_r - p_l

            # 特徵變量
            d_alpha_1 = (d_p - rho_roe * a_roe * d_vx) / (2 * a_sq)
            d_alpha_2 = d_rho - d_p / a_sq
            d_alpha_3 = (d_p + rho_roe * a_roe * d_vx) / (2 * a_sq)

            # 左右通量
            # 左狀態通量
            f_mass_l = rho_l * vx_l
            f_momx_l = rho_l * vx_l**2 + p_l
            f_momr_l = rho_l * vx_l * vr_l
            f_momt_l = rho_l * vx_l * vt_l
            f_energy_l = rho_l * vx_l * ho_l

            # 右狀態通量
            f_mass_r = rho_r * vx_r
            f_momx_r = rho_r * vx_r**2 + p_r
            f_momr_r = rho_r * vx_r * vr_r
            f_momt_r = rho_r * vx_r * vt_r
            f_energy_r = rho_r * vx_r * ho_r

            # Roe 通量 = 0.5*(F_L + F_R) - 0.5*|A|*(U_R - U_L)
            # 耗散項
            diss_mass = lambda_1 * d_alpha_1 + lambda_2 * d_alpha_2 + lambda_3 * d_alpha_3
            diss_momx = (
                lambda_1 * d_alpha_1 * (vx_roe - a_roe)
                + lambda_2 * d_alpha_2 * vx_roe
                + lambda_3 * d_alpha_3 * (vx_roe + a_roe)
            )
            diss_momr = (
                lambda_1 * d_alpha_1 * vr_roe
                + lambda_2 * (d_alpha_2 * vr_roe + rho_roe * d_vr)
                + lambda_3 * d_alpha_3 * vr_roe
            )
            diss_momt = (
                lambda_1 * d_alpha_1 * vt_roe
                + lambda_2 * (d_alpha_2 * vt_roe + rho_roe * d_vt)
                + lambda_3 * d_alpha_3 * vt_roe
            )
            diss_energy = (
                lambda_1 * d_alpha_1 * (ho_roe - vx_roe * a_roe)
                + lambda_2 * (d_alpha_2 * 0.5 * v_sq + rho_roe * (vr_roe * d_vr + vt_roe * d_vt))
                + lambda_3 * d_alpha_3 * (ho_roe + vx_roe * a_roe)
            )

            # 最終通量
            flux_mass[:, j, :] = 0.5 * (f_mass_l + f_mass_r) - 0.5 * diss_mass
            flux_momx[:, j, :] = 0.5 * (f_momx_l + f_momx_r) - 0.5 * diss_momx
            flux_momr[:, j, :] = 0.5 * (f_momr_l + f_momr_r) - 0.5 * diss_momr
            flux_momt[:, j, :] = 0.5 * (f_momt_l + f_momt_r) - 0.5 * diss_momt
            flux_energy[:, j, :] = 0.5 * (f_energy_l + f_energy_r) - 0.5 * diss_energy

        return flux_mass, flux_momx, flux_momr, flux_momt, flux_energy

    def compute_convective_flux_theta(
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
        """計算 θ 方向（周向）對流通量。

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            (質量通量, X動量通量, R動量通量, θ動量通量, 能量通量)
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 初始化通量數組
        flux_mass = np.zeros((im + 1, jm, km))
        flux_momx = np.zeros((im + 1, jm, km))
        flux_momr = np.zeros((im + 1, jm, km))
        flux_momt = np.zeros((im + 1, jm, km))
        flux_energy = np.zeros((im + 1, jm, km))

        # 對每個內部面計算通量（簡化版本，使用中心差分）
        for i in range(1, im):
            # 左右狀態
            rho_l = flow.rho[i - 1, :, :]
            rho_r = flow.rho[i, :, :]

            vt_l = flow.vt[i - 1, :, :]
            vt_r = flow.vt[i, :, :]

            p_l = flow.p[i - 1, :, :]
            p_r = flow.p[i, :, :]

            ho_l = flow.ho[i - 1, :, :]
            ho_r = flow.ho[i, :, :]

            # 平均狀態
            rho_avg = 0.5 * (rho_l + rho_r)
            vx_avg = 0.5 * (flow.vx[i - 1, :, :] + flow.vx[i, :, :])
            vr_avg = 0.5 * (flow.vr[i - 1, :, :] + flow.vr[i, :, :])
            vt_avg = 0.5 * (vt_l + vt_r)
            p_avg = 0.5 * (p_l + p_r)
            ho_avg = 0.5 * (ho_l + ho_r)

            # 通量（中心差分）
            flux_mass[i, :, :] = rho_avg * vt_avg
            flux_momx[i, :, :] = rho_avg * vt_avg * vx_avg
            flux_momr[i, :, :] = rho_avg * vt_avg * vr_avg
            flux_momt[i, :, :] = rho_avg * vt_avg**2 + p_avg
            flux_energy[i, :, :] = rho_avg * vt_avg * ho_avg

        # 周期性邊界
        flux_mass[0, :, :] = flux_mass[im - 1, :, :]
        flux_mass[im, :, :] = flux_mass[1, :, :]
        flux_momx[0, :, :] = flux_momx[im - 1, :, :]
        flux_momx[im, :, :] = flux_momx[1, :, :]
        flux_momr[0, :, :] = flux_momr[im - 1, :, :]
        flux_momr[im, :, :] = flux_momr[1, :, :]
        flux_momt[0, :, :] = flux_momt[im - 1, :, :]
        flux_momt[im, :, :] = flux_momt[1, :, :]
        flux_energy[0, :, :] = flux_energy[im - 1, :, :]
        flux_energy[im, :, :] = flux_energy[1, :, :]

        return flux_mass, flux_momx, flux_momr, flux_momt, flux_energy

    def compute_convective_flux_r(
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
        """計算 R 方向（徑向）對流通量。

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            (質量通量, X動量通量, R動量通量, θ動量通量, 能量通量)
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 初始化通量數組
        flux_mass = np.zeros((im, jm, km + 1))
        flux_momx = np.zeros((im, jm, km + 1))
        flux_momr = np.zeros((im, jm, km + 1))
        flux_momt = np.zeros((im, jm, km + 1))
        flux_energy = np.zeros((im, jm, km + 1))

        # 對每個內部面計算通量
        for k in range(1, km):
            # 下上狀態
            rho_d = flow.rho[:, :, k - 1]
            rho_u = flow.rho[:, :, k]

            vr_d = flow.vr[:, :, k - 1]
            vr_u = flow.vr[:, :, k]

            p_d = flow.p[:, :, k - 1]
            p_u = flow.p[:, :, k]

            ho_d = flow.ho[:, :, k - 1]
            ho_u = flow.ho[:, :, k]

            # 平均狀態
            rho_avg = 0.5 * (rho_d + rho_u)
            vx_avg = 0.5 * (flow.vx[:, :, k - 1] + flow.vx[:, :, k])
            vr_avg = 0.5 * (vr_d + vr_u)
            vt_avg = 0.5 * (flow.vt[:, :, k - 1] + flow.vt[:, :, k])
            p_avg = 0.5 * (p_d + p_u)
            ho_avg = 0.5 * (ho_d + ho_u)

            # 通量
            flux_mass[:, :, k] = rho_avg * vr_avg
            flux_momx[:, :, k] = rho_avg * vr_avg * vx_avg
            flux_momr[:, :, k] = rho_avg * vr_avg**2 + p_avg
            flux_momt[:, :, k] = rho_avg * vr_avg * vt_avg
            flux_energy[:, :, k] = rho_avg * vr_avg * ho_avg

        # 壁面邊界（無穿透）
        flux_mass[:, :, 0] = 0.0
        flux_mass[:, :, km] = 0.0
        flux_momr[:, :, 0] = flow.p[:, :, 0]  # 壓力
        flux_momr[:, :, km] = flow.p[:, :, km - 1]

        return flux_mass, flux_momx, flux_momr, flux_momt, flux_energy

    def compute_residual(
        self,
        flow: FlowField,
        grid: Grid3D,
        flux_x: tuple[NDArray[np.float64], ...],
        flux_theta: tuple[NDArray[np.float64], ...],
        flux_r: tuple[NDArray[np.float64], ...],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """計算殘差（通量散度）。

        R = ∂F_x/∂x + ∂F_θ/(r∂θ) + ∂F_r/∂r + 源項

        Args:
            flow: 流場數據
            grid: 網格數據
            flux_x: X 方向通量
            flux_theta: θ 方向通量
            flux_r: R 方向通量

        Returns:
            各守恆量的殘差
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 初始化殘差
        res_mass = np.zeros((im, jm, km))
        res_momx = np.zeros((im, jm, km))
        res_momr = np.zeros((im, jm, km))
        res_momt = np.zeros((im, jm, km))
        res_energy = np.zeros((im, jm, km))

        # 假設均勻網格間距（實際應從 grid 獲取）
        dx = 0.001
        dtheta = 2.0 * np.pi / (im - 1)
        dr = 0.01

        # X 方向通量散度
        for j in range(jm):
            res_mass[:, j, :] += (flux_x[0][:, j + 1, :] - flux_x[0][:, j, :]) / dx
            res_momx[:, j, :] += (flux_x[1][:, j + 1, :] - flux_x[1][:, j, :]) / dx
            res_momr[:, j, :] += (flux_x[2][:, j + 1, :] - flux_x[2][:, j, :]) / dx
            res_momt[:, j, :] += (flux_x[3][:, j + 1, :] - flux_x[3][:, j, :]) / dx
            res_energy[:, j, :] += (flux_x[4][:, j + 1, :] - flux_x[4][:, j, :]) / dx

        # θ 方向通量散度（需要考慮 1/r）
        r_avg = 0.3  # 平均半徑，實際應從 grid 獲取
        for i in range(im):
            res_mass[i, :, :] += (flux_theta[0][i + 1, :, :] - flux_theta[0][i, :, :]) / (
                r_avg * dtheta
            )
            res_momx[i, :, :] += (flux_theta[1][i + 1, :, :] - flux_theta[1][i, :, :]) / (
                r_avg * dtheta
            )
            res_momr[i, :, :] += (flux_theta[2][i + 1, :, :] - flux_theta[2][i, :, :]) / (
                r_avg * dtheta
            )
            res_momt[i, :, :] += (flux_theta[3][i + 1, :, :] - flux_theta[3][i, :, :]) / (
                r_avg * dtheta
            )
            res_energy[i, :, :] += (flux_theta[4][i + 1, :, :] - flux_theta[4][i, :, :]) / (
                r_avg * dtheta
            )

        # R 方向通量散度
        for k in range(km):
            res_mass[:, :, k] += (flux_r[0][:, :, k + 1] - flux_r[0][:, :, k]) / dr
            res_momx[:, :, k] += (flux_r[1][:, :, k + 1] - flux_r[1][:, :, k]) / dr
            res_momr[:, :, k] += (flux_r[2][:, :, k + 1] - flux_r[2][:, :, k]) / dr
            res_momt[:, :, k] += (flux_r[3][:, :, k + 1] - flux_r[3][:, :, k]) / dr
            res_energy[:, :, k] += (flux_r[4][:, :, k + 1] - flux_r[4][:, :, k]) / dr

        # 源項（離心力、科氏力）
        # TODO: 添加源項計算

        return res_mass, res_momx, res_momr, res_momt, res_energy


class ArtificialViscosity:
    """人工黏性計算。

    用於穩定數值格式和捕捉激波。
    """

    def __init__(
        self,
        sf_2nd: float = 0.005,
        sf_4th: float = 0.8,
    ) -> None:
        """初始化人工黏性。

        Args:
            sf_2nd: 二階人工黏性係數
            sf_4th: 四階人工黏性比例
        """
        self.sf_2nd = sf_2nd
        self.sf_4th = sf_4th

    def compute_pressure_sensor(
        self,
        p: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """計算壓力感測器（用於激波檢測）。

        ν = |p_{i+1} - 2p_i + p_{i-1}| / (p_{i+1} + 2p_i + p_{i-1})

        Args:
            p: 壓力場

        Returns:
            壓力感測器值
        """
        im, jm, km = p.shape

        sensor = np.zeros_like(p)

        # J 方向感測器
        for j in range(1, jm - 1):
            dp2 = np.abs(p[:, j + 1, :] - 2.0 * p[:, j, :] + p[:, j - 1, :])
            sp = p[:, j + 1, :] + 2.0 * p[:, j, :] + p[:, j - 1, :]
            sensor[:, j, :] = np.maximum(sensor[:, j, :], dp2 / np.maximum(sp, 1e-10))

        return sensor

    def compute_artificial_dissipation(
        self,
        flow: FlowField,
        direction: str = "x",
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """計算人工耗散。

        D = ε^(2) * Δ - ε^(4) * Δ^3

        Args:
            flow: 流場數據
            direction: 方向 ('x', 'theta', 'r')

        Returns:
            各守恆量的人工耗散
        """
        im, jm, km = flow.im, flow.jm, flow.km

        # 初始化耗散
        diss_mass = np.zeros((im, jm, km))
        diss_momx = np.zeros((im, jm, km))
        diss_momr = np.zeros((im, jm, km))
        diss_momt = np.zeros((im, jm, km))
        diss_energy = np.zeros((im, jm, km))

        # 壓力感測器
        sensor = self.compute_pressure_sensor(flow.p)

        # 二階和四階係數
        # ε^(2) 在激波附近激活（通過壓力感測器）
        # ε^(4) 提供背景耗散，但在激波附近減小
        eps2 = self.sf_2nd * sensor
        eps4 = np.maximum(0.0, self.sf_4th - eps2)

        if direction == "x":
            # J 方向耗散
            # 二階耗散：Δ^2 = (U_{j+1} - 2*U_j + U_{j-1})
            # 四階耗散：Δ^4 = (U_{j+2} - 4*U_{j+1} + 6*U_j - 4*U_{j-1} + U_{j-2})

            # 二階耗散 (j = 1 到 jm-2)
            for j in range(1, jm - 1):
                eps2_face = 0.5 * (eps2[:, j, :] + eps2[:, j + 1, :])
                d2_rho = flow.rho[:, j + 1, :] - 2.0 * flow.rho[:, j, :] + flow.rho[:, j - 1, :]
                diss_mass[:, j, :] = eps2_face * d2_rho

                d2_rovx = flow.rovx[:, j + 1, :] - 2.0 * flow.rovx[:, j, :] + flow.rovx[:, j - 1, :]
                diss_momx[:, j, :] = eps2_face * d2_rovx

                d2_rovr = flow.rovr[:, j + 1, :] - 2.0 * flow.rovr[:, j, :] + flow.rovr[:, j - 1, :]
                diss_momr[:, j, :] = eps2_face * d2_rovr

                d2_rorvt = (
                    flow.rorvt[:, j + 1, :] - 2.0 * flow.rorvt[:, j, :] + flow.rorvt[:, j - 1, :]
                )
                diss_momt[:, j, :] = eps2_face * d2_rorvt

                d2_roe = flow.roe[:, j + 1, :] - 2.0 * flow.roe[:, j, :] + flow.roe[:, j - 1, :]
                diss_energy[:, j, :] = eps2_face * d2_roe

            # 四階耗散 (j = 2 到 jm-3，需要更多鄰點)
            for j in range(2, jm - 2):
                eps4_face = 0.5 * (eps4[:, j, :] + eps4[:, j + 1, :])

                # 四階差分：Δ^4 = U_{j+2} - 4*U_{j+1} + 6*U_j - 4*U_{j-1} + U_{j-2}
                d4_rho = (
                    flow.rho[:, j + 2, :]
                    - 4.0 * flow.rho[:, j + 1, :]
                    + 6.0 * flow.rho[:, j, :]
                    - 4.0 * flow.rho[:, j - 1, :]
                    + flow.rho[:, j - 2, :]
                )
                diss_mass[:, j, :] -= eps4_face * d4_rho

                d4_rovx = (
                    flow.rovx[:, j + 2, :]
                    - 4.0 * flow.rovx[:, j + 1, :]
                    + 6.0 * flow.rovx[:, j, :]
                    - 4.0 * flow.rovx[:, j - 1, :]
                    + flow.rovx[:, j - 2, :]
                )
                diss_momx[:, j, :] -= eps4_face * d4_rovx

                d4_rovr = (
                    flow.rovr[:, j + 2, :]
                    - 4.0 * flow.rovr[:, j + 1, :]
                    + 6.0 * flow.rovr[:, j, :]
                    - 4.0 * flow.rovr[:, j - 1, :]
                    + flow.rovr[:, j - 2, :]
                )
                diss_momr[:, j, :] -= eps4_face * d4_rovr

                d4_rorvt = (
                    flow.rorvt[:, j + 2, :]
                    - 4.0 * flow.rorvt[:, j + 1, :]
                    + 6.0 * flow.rorvt[:, j, :]
                    - 4.0 * flow.rorvt[:, j - 1, :]
                    + flow.rorvt[:, j - 2, :]
                )
                diss_momt[:, j, :] -= eps4_face * d4_rorvt

                d4_roe = (
                    flow.roe[:, j + 2, :]
                    - 4.0 * flow.roe[:, j + 1, :]
                    + 6.0 * flow.roe[:, j, :]
                    - 4.0 * flow.roe[:, j - 1, :]
                    + flow.roe[:, j - 2, :]
                )
                diss_energy[:, j, :] -= eps4_face * d4_roe

        return diss_mass, diss_momx, diss_momr, diss_momt, diss_energy
