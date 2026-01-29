# -*- coding: utf-8 -*-
"""速度三角形計算模組。

提供各種輸入方法的速度三角形計算功能。
"""

from __future__ import annotations

import math

from .data_structures import CONSTANTS, MachineType, VelocityTriangle


class VelocityTriangleCalculator:
    """速度三角形計算器。"""

    def __init__(self, machine_type: MachineType) -> None:
        """初始化速度三角形計算器。

        Args:
            machine_type: 機械類型（渦輪或壓縮機）
        """
        self.machine_type = machine_type
        self.deg2rad = CONSTANTS["DEG2RAD"]
        self.rad2deg = CONSTANTS["RAD2DEG"]

    def calculate_type_a(
        self, phi: float, psi: float, reaction: float, u: float
    ) -> tuple[float, float, float, float]:
        """使用反應度/流量係數/負荷係數法計算速度三角形（Type A）。

        Args:
            phi: 流量係數
            psi: 負荷係數
            reaction: 反應度
            u: 圓周速度 [m/s]

        Returns:
            (alpha_in, alpha_out, beta_in, beta_out) 全部為角度
        """
        if self.machine_type == MachineType.TURBINE:
            # 渦輪速度三角形
            # φ = 2(1-λ)/(tan(β₁) + tan(β₂))
            # ψ = 2(1-λ-φ·tan(α₀))

            # 求解絕對出口角 α₀
            # ψ = 2(1 - λ - φ·tan(α₀))
            # tan(α₀) = (2(1-λ) - ψ) / (2φ)
            tan_alpha_out = (2.0 * (1.0 - reaction) - psi) / (2.0 * phi)
            alpha_out = math.atan(tan_alpha_out) * self.rad2deg

            # 對於軸向重複級，進口角等於出口角
            alpha_in = alpha_out

            # 從流量係數計算轉子角度
            # φ = 2(1-λ)/(tan(β₁) + tan(β₂))
            # 假設 β₁ = β₂（對稱設計的簡化）
            # tan(β₁) = 2(1-λ)/φ / 2 = (1-λ)/φ
            tan_beta = (1.0 - reaction) / phi
            beta_in = math.atan(tan_beta) * self.rad2deg

            # 從速度三角形關係計算出口相對角
            # tan(β₂) = tan(α₂) - 1/φ
            tan_beta_out = tan_alpha_out - 1.0 / phi
            beta_out = math.atan(tan_beta_out) * self.rad2deg

        else:  # COMPRESSOR
            # 壓縮機速度三角形
            # φ = -2λ/(tan(β₁) + tan(β₂))
            # ψ = -2(1-λ-φ·tan(α₂))

            # 求解絕對出口角 α₂
            # ψ = -2(1 - λ - φ·tan(α₂))
            # tan(α₂) = (1 - λ + ψ/2) / φ
            tan_alpha_out = (1.0 - reaction + psi / 2.0) / phi
            alpha_out = math.atan(tan_alpha_out) * self.rad2deg

            # 進口角（軸向）
            alpha_in = 0.0  # 軸向進入

            # 從流量係數計算轉子進口角
            # φ = -2λ/(tan(β₁) + tan(β₂))
            # 簡化：假設對稱
            tan_beta_in = -reaction / phi
            beta_in = math.atan(tan_beta_in) * self.rad2deg

            # 出口相對角
            # tan(β₂) = tan(α₂) + 1/φ
            tan_beta_out = tan_alpha_out + 1.0 / phi
            beta_out = math.atan(tan_beta_out) * self.rad2deg

        return alpha_in, alpha_out, beta_in, beta_out

    def calculate_type_b(
        self, phi: float, alpha_2: float, beta_1: float
    ) -> tuple[float, float, float, float]:
        """使用流量/定子角/轉子角法計算速度三角形（Type B）。

        Args:
            phi: 流量係數
            alpha_2: 定子出口絕對角 [度]
            beta_1: 轉子進口相對角 [度]

        Returns:
            (alpha_in, alpha_out, beta_in, beta_out) 全部為角度
        """
        # 轉換為弧度
        alpha_2_rad = alpha_2 * self.deg2rad
        beta_1_rad = beta_1 * self.deg2rad

        tan_alpha_2 = math.tan(alpha_2_rad)
        tan_beta_1 = math.tan(beta_1_rad)

        if self.machine_type == MachineType.TURBINE:
            # 渦輪
            # tan(β₂) = tan(α₂) - 1/φ
            tan_beta_2 = tan_alpha_2 - 1.0 / phi

            # tan(α₁) = tan(β₁) + 1/φ
            tan_alpha_1 = tan_beta_1 + 1.0 / phi

            alpha_in = math.atan(tan_alpha_1) * self.rad2deg
            alpha_out = alpha_2
            beta_in = beta_1
            beta_out = math.atan(tan_beta_2) * self.rad2deg

        else:  # COMPRESSOR
            # 壓縮機
            # tan(β₂) = tan(α₂) + 1/φ
            tan_beta_2 = tan_alpha_2 + 1.0 / phi

            # tan(α₁) = tan(β₁) - 1/φ
            tan_alpha_1 = tan_beta_1 - 1.0 / phi

            alpha_in = math.atan(tan_alpha_1) * self.rad2deg
            alpha_out = alpha_2
            beta_in = beta_1
            beta_out = math.atan(tan_beta_2) * self.rad2deg

        return alpha_in, alpha_out, beta_in, beta_out

    def create_velocity_triangle(
        self,
        u: float,
        vm: float,
        alpha: float,
        is_rotor: bool = False,
    ) -> VelocityTriangle:
        """從基本參數創建速度三角形。

        Args:
            u: 圓周速度 [m/s]
            vm: 子午速度 [m/s]
            alpha: 絕對流角 [度]
            is_rotor: 是否為轉子（計算相對速度）

        Returns:
            速度三角形
        """
        alpha_rad = alpha * self.deg2rad

        # 絕對速度分量
        vtheta = vm * math.tan(alpha_rad)

        # 相對速度（對於轉子）
        if is_rotor:
            wtheta = vtheta - u
            beta = math.atan2(wtheta, vm) * self.rad2deg
        else:
            beta = alpha

        # 需要聲速來計算馬赫數（這裡設置為0，由調用者更新）
        mach_abs = 0.0
        mach_rel = 0.0

        return VelocityTriangle(
            vm=vm,
            vtheta=vtheta,
            u=u,
            alpha=alpha,
            beta=beta,
            mach_abs=mach_abs,
            mach_rel=mach_rel,
        )

    def apply_free_vortex(
        self, vt_design: VelocityTriangle, r_design: float, r_local: float, frac_twist: float
    ) -> VelocityTriangle:
        """應用自由渦設計調整速度三角形。

        Args:
            vt_design: 設計點速度三角形
            r_design: 設計半徑 [m]
            r_local: 局部半徑 [m]
            frac_twist: 扭轉比例 (0=無扭轉, 1=完全自由渦)

        Returns:
            局部速度三角形
        """
        # 半徑比
        r_ratio = r_design / r_local

        # 自由渦條件：r·Vθ = constant
        # Vθ(r) = Vθ_design × (r_design/r)
        vtheta_fv = vt_design.vtheta * r_ratio

        # 實際切向速度（考慮扭轉比例）
        vtheta = vt_design.vtheta * (1.0 - frac_twist) + vtheta_fv * frac_twist

        # 圓周速度變化
        u_local = vt_design.u * r_local / r_design

        # 子午速度（假設不變，實際可能有變化）
        vm = vt_design.vm

        # 計算角度
        alpha = math.atan2(vtheta, vm) * self.rad2deg

        # 相對速度
        wtheta = vtheta - u_local
        beta = math.atan2(wtheta, vm) * self.rad2deg

        return VelocityTriangle(
            vm=vm,
            vtheta=vtheta,
            u=u_local,
            alpha=alpha,
            beta=beta,
            mach_abs=vt_design.mach_abs,
            mach_rel=vt_design.mach_rel,
        )

    def calculate_flow_coefficient(self, vm: float, u: float) -> float:
        """計算流量係數。

        Args:
            vm: 子午速度 [m/s]
            u: 圓周速度 [m/s]

        Returns:
            流量係數 φ = Vm/U
        """
        return vm / u if u > 0 else 0.0

    def calculate_loading_coefficient(self, dh: float, u: float) -> float:
        """計算負荷係數。

        Args:
            dh: 焓變 [J/kg]
            u: 圓周速度 [m/s]

        Returns:
            負荷係數 ψ = ΔH/U²
        """
        return dh / (u * u) if u > 0 else 0.0

    def calculate_reaction(
        self, dh_rotor: float, dh_stage: float
    ) -> float:
        """計算反應度。

        Args:
            dh_rotor: 轉子焓變 [J/kg]
            dh_stage: 級總焓變 [J/kg]

        Returns:
            反應度 λ = ΔH_rotor/ΔH_stage
        """
        return dh_rotor / dh_stage if dh_stage != 0 else 0.5
