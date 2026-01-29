# -*- coding: utf-8 -*-
"""MULTALL 逆向設計模組。

提供葉片逆向設計功能：
- 基於目標壓力分佈設計葉片幾何
- 迭代修正葉片形狀以匹配設計目標
- 計算相容出口角度和葉片數

基於 FORTRAN RE_DESIGN 和 GEOM_MOD 子程序移植。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties


class InverseDesignType(IntEnum):
    """逆向設計類型。"""

    PRESSURE_LOADING = 1  # 基於壓力載荷
    EXIT_ANGLE = 2  # 基於出口角度
    BLADE_FORCE = 3  # 基於葉片力


@dataclass
class InverseDesignParameters:
    """逆向設計參數。"""

    enabled: bool = False
    design_type: InverseDesignType = InverseDesignType.PRESSURE_LOADING

    # 迭代控制
    max_iterations: int = 100
    convergence_tolerance: float = 1e-4

    # 鬆弛因子
    angle_relaxation: float = 0.1  # 角度更新鬆弛因子
    rotation_relaxation: float = 0.5  # 旋轉更新鬆弛因子
    thickness_relaxation: float = 0.3  # 厚度更新鬆弛因子

    # 目標條件
    target_exit_angle: float = 0.0  # 目標出口角度 [rad]
    target_blade_force: float = 0.0  # 目標葉片切向力 [N]

    # 設計範圍
    j_leading_edge: int = 0  # 前緣 J 索引
    j_trailing_edge: int = 0  # 後緣 J 索引


@dataclass
class BladeDesignSection:
    """葉片設計截面。"""

    k_index: int  # 跨向索引

    # 流線面座標
    x_stream: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 軸向座標
    r_stream: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 半徑座標
    s_merid: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 子午線距離

    # 葉片幾何
    frac_chord: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 弦長分數
    beta_camber: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 中弧線角度 [rad]
    thick_upper: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 上表面厚度
    thick_lower: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # 下表面厚度

    # 輸出座標
    rt_upper: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # R*θ 上表面
    rt_thickness: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )  # R*θ 厚度


@dataclass
class InverseDesignResult:
    """逆向設計結果。"""

    converged: bool = False
    iterations: int = 0

    # 當前流動條件
    current_exit_angle: float = 0.0  # 當前出口角度 [rad]
    current_blade_force: float = 0.0  # 當前葉片力 [N]

    # 相容條件
    compatible_exit_angle: float = 0.0  # 相容出口角度 [rad]
    compatible_blade_number: int = 0  # 相容葉片數

    # 設計修正
    angle_correction: float = 0.0  # 角度修正量 [rad]
    rotation_angle: float = 0.0  # 葉片旋轉角度 [rad]


class BladeRedesigner:
    """葉片重新設計器。

    基於目標中弧線和厚度分佈生成新的葉片幾何。
    """

    def __init__(self, smooth_factor: float = 0.25, smooth_iterations: int = 3):
        """初始化。

        Args:
            smooth_factor: 平滑因子
            smooth_iterations: 平滑迭代次數
        """
        self.smooth_factor = smooth_factor
        self.smooth_iterations = smooth_iterations

    def create_section(
        self,
        k_index: int,
        x_stream: NDArray[np.float64],
        r_stream: NDArray[np.float64],
        relative_spacing: NDArray[np.float64],
        n_le: int,
        n_te: int,
        frac_new: NDArray[np.float64],
        beta_new: NDArray[np.float64],
        thick_upper_frac: NDArray[np.float64],
        thick_lower_frac: NDArray[np.float64],
        frac_chord_up: float = 0.2,
        frac_chord_down: float = 0.2,
        rtheta_mid: float = 0.0,
    ) -> BladeDesignSection:
        """創建新的葉片截面。

        Args:
            k_index: 跨向索引
            x_stream: 流線面軸向座標
            r_stream: 流線面半徑座標
            relative_spacing: 相對網格間距
            n_le: 前緣點索引
            n_te: 後緣點索引
            frac_new: 新的弦長分數分佈
            beta_new: 新的中弧線角度分佈 [deg]
            thick_upper_frac: 上表面厚度（弦長分數）
            thick_lower_frac: 下表面厚度（弦長分數）
            frac_chord_up: 上游延伸長度（弦長分數）
            frac_chord_down: 下游延伸長度（弦長分數）
            rtheta_mid: 中點 R*θ 值

        Returns:
            新的葉片設計截面
        """
        section = BladeDesignSection(k_index=k_index)

        # 計算流線面子午線距離
        n_ss = len(x_stream)
        s_ss = np.zeros(n_ss)
        for n in range(1, n_ss):
            dx = x_stream[n] - x_stream[n - 1]
            dr = r_stream[n] - r_stream[n - 1]
            s_ss[n] = s_ss[n - 1] + np.sqrt(dx * dx + dr * dr)

        # 子午線弦長
        s_merd = s_ss[n_te] - s_ss[n_le]

        # 平滑輸入數據
        beta_smooth = self._smooth_data(frac_new, beta_new)
        thick_up_smooth = self._smooth_data(frac_new, thick_upper_frac)
        thick_low_smooth = self._smooth_data(frac_new, thick_lower_frac)

        # 按弦長縮放厚度
        thick_upper = thick_up_smooth * s_merd
        thick_lower = thick_low_smooth * s_merd

        # 無量綱化子午線距離
        s_rel = (s_ss - s_ss[n_le]) / s_merd

        # 計算網格點的弦長分數
        jm_row = n_te - n_le + 1
        j_le_row = n_le
        j_te_row = n_te

        # 插值得到網格點的相對間距
        spacing = np.zeros(jm_row + 1)
        for j in range(jm_row + 1):
            frac_j = j / (j_te_row - j_le_row)
            spacing[j] = np.interp(frac_j, s_rel, relative_spacing)

        # 計算弦長分數
        frac_chord = np.zeros(jm_row + 1)
        for j in range(1, jm_row + 1):
            frac_chord[j] = frac_chord[j - 1] + 0.5 * (spacing[j] + spacing[j - 1])

        # 歸一化
        if frac_chord[-1] > 0:
            frac_chord = frac_chord / frac_chord[-1]

        # 設置最終子午線位置
        s_dist = s_ss[n_le] + s_merd * frac_chord

        # 插值獲取 X 和 R 座標
        x_new = np.interp(s_dist, s_ss, x_stream)
        r_new = np.interp(s_dist, s_ss, r_stream)

        # 插值獲取葉片參數
        frac_s = (s_dist - s_dist[0]) / (s_dist[-1] - s_dist[0])
        beta_camber = np.interp(frac_s, frac_new, beta_smooth)
        thick_up = np.interp(frac_s, frac_new, thick_upper)
        thick_low = np.interp(frac_s, frac_new, thick_lower)

        # 計算 R*θ 座標
        rt_upper = np.zeros(len(x_new))
        rt_thickness = np.zeros(len(x_new))

        # 葉片上的網格
        deg_to_rad = np.pi / 180.0
        theta_mid = 0.0
        for j in range(1, len(x_new)):
            dthdx = np.tan(beta_camber[j] * deg_to_rad) / r_new[j]
            dthdx_prev = np.tan(beta_camber[j - 1] * deg_to_rad) / r_new[j - 1]
            theta_mid += 0.5 * (dthdx + dthdx_prev) * (s_dist[j] - s_dist[j - 1])
            rt_upper[j] = theta_mid * r_new[j] + thick_up[j]
            rt_thickness[j] = thick_up[j] + thick_low[j]

        # 設置 R*θ_mid 在中點
        j_mid = len(x_new) // 2
        theta_shift = (
            rtheta_mid - (rt_upper[j_mid] - 0.5 * rt_thickness[j_mid])
        ) / r_new[j_mid]
        rt_upper = rt_upper + theta_shift * r_new

        # 存儲結果
        section.x_stream = x_new
        section.r_stream = r_new
        section.s_merid = s_dist
        section.frac_chord = frac_chord
        section.beta_camber = beta_camber
        section.thick_upper = thick_up
        section.thick_lower = thick_low
        section.rt_upper = rt_upper
        section.rt_thickness = rt_thickness

        return section

    def _smooth_data(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """平滑數據。

        Args:
            x: X 座標
            y: Y 值

        Returns:
            平滑後的 Y 值
        """
        y_smooth = y.copy()
        n = len(y)

        for _ in range(self.smooth_iterations):
            y_new = y_smooth.copy()
            for i in range(1, n - 1):
                y_new[i] = (1 - 2 * self.smooth_factor) * y_smooth[i] + self.smooth_factor * (
                    y_smooth[i - 1] + y_smooth[i + 1]
                )
            y_smooth = y_new

        return y_smooth


class InverseDesignSolver:
    """逆向設計求解器。

    迭代修正葉片幾何以達到設計目標。
    """

    def __init__(
        self, gas: GasProperties, params: InverseDesignParameters | None = None
    ):
        """初始化。

        Args:
            gas: 氣體性質
            params: 逆向設計參數
        """
        self.gas = gas
        self.params = params or InverseDesignParameters()
        self.redesigner = BladeRedesigner()

        # 設計歷史
        self.history: list[InverseDesignResult] = []

    def compute_current_conditions(
        self,
        flow: FlowField,
        j_le: int,
        j_te: int,
        omega: float = 0.0,
        n_blades: int = 1,
    ) -> dict[str, float]:
        """計算當前流動條件。

        Args:
            flow: 流場
            j_le: 前緣 J 索引
            j_te: 後緣 J 索引
            omega: 角速度 [rad/s]
            n_blades: 葉片數

        Returns:
            包含當前條件的字典
        """
        jm = flow.jm

        # 入口和出口平均
        j_in = 1
        j_out = jm - 2

        # 質量流量加權平均
        vx_in = np.mean(flow.vx[:, j_in, :])
        vx_out = np.mean(flow.vx[:, j_out, :])
        vr_out = np.mean(flow.vr[:, j_out, :])
        vt_in = np.mean(flow.vt[:, j_in, :])
        vt_out = np.mean(flow.vt[:, j_out, :])
        rho_out = np.mean(flow.rho[:, j_out, :])

        # 子午線速度
        vm_out = np.sqrt(vx_out**2 + vr_out**2)

        # 相對速度（轉子坐標系）
        # 假設平均半徑
        r_avg_out = 0.5  # 需要從網格獲取
        u_blade = omega * r_avg_out
        wt_out = vt_out - u_blade

        # 相對出口角度
        if vm_out > 1e-10:
            exit_angle = np.arctan(wt_out / vm_out)
        else:
            exit_angle = 0.0

        # 計算葉片切向力（壓力差積分）
        # 簡化：使用入口出口動量差
        blade_force = (
            rho_out * vm_out * (vt_out - vt_in) * 2 * np.pi * r_avg_out / n_blades
        )

        return {
            "vx_in": vx_in,
            "vx_out": vx_out,
            "vt_in": vt_in,
            "vt_out": vt_out,
            "vm_out": vm_out,
            "exit_angle": exit_angle,
            "blade_force": blade_force,
            "rho_out": rho_out,
        }

    def compute_compatible_angle(
        self,
        current_conditions: dict[str, float],
        target_force: float,
        omega: float = 0.0,
        r_avg: float = 0.5,
    ) -> float:
        """計算相容出口角度。

        基於目標葉片力計算需要的出口角度。

        Args:
            current_conditions: 當前流動條件
            target_force: 目標葉片力 [N]
            omega: 角速度 [rad/s]
            r_avg: 平均半徑 [m]

        Returns:
            相容出口角度 [rad]
        """
        rho = current_conditions["rho_out"]
        vm_out = current_conditions["vm_out"]
        vt_in = current_conditions["vt_in"]

        u_blade = omega * r_avg
        w_out_sq = vm_out**2 + (current_conditions["vt_out"] - u_blade) ** 2

        if w_out_sq < 1e-10 or rho < 1e-10:
            return 0.0

        # 力係數
        d_force = target_force / (rho * r_avg * w_out_sq)

        # 迭代求解相容角度
        angle = current_conditions["exit_angle"]
        w_rel_out = np.sqrt(w_out_sq)

        for _ in range(5):
            sin_a = np.sin(angle)
            denominator = u_blade / w_rel_out + sin_a - r_avg * vt_in / (
                r_avg * w_rel_out
            )
            if abs(denominator) > 1e-10:
                cos_a_new = d_force / denominator
                cos_a_new = np.clip(cos_a_new, -0.999, 0.999)
                sin_a_new = np.sqrt(1 - cos_a_new**2) * np.sign(angle + 1e-6)
                sin_a = 0.9 * sin_a + 0.1 * sin_a_new

        angle_compatible = np.arcsin(np.clip(sin_a, -1, 1))
        return angle_compatible

    def compute_rotation_angle(
        self, current_angle: float, target_angle: float
    ) -> float:
        """計算葉片旋轉角度。

        Args:
            current_angle: 當前出口角度 [rad]
            target_angle: 目標出口角度 [rad]

        Returns:
            葉片旋轉角度 [rad]
        """
        return (target_angle - current_angle) * self.params.rotation_relaxation

    def iterate(
        self,
        flow: FlowField,
        target_pressure_ps: NDArray[np.float64] | None = None,
        target_pressure_ss: NDArray[np.float64] | None = None,
        omega: float = 0.0,
        n_blades: int = 1,
    ) -> InverseDesignResult:
        """執行一次逆向設計迭代。

        Args:
            flow: 流場
            target_pressure_ps: 目標壓力面壓力分佈
            target_pressure_ss: 目標吸力面壓力分佈
            omega: 角速度 [rad/s]
            n_blades: 葉片數

        Returns:
            逆向設計結果
        """
        result = InverseDesignResult()

        j_le = self.params.j_leading_edge
        j_te = self.params.j_trailing_edge

        # 計算當前條件
        conditions = self.compute_current_conditions(
            flow, j_le, j_te, omega, n_blades
        )

        result.current_exit_angle = conditions["exit_angle"]
        result.current_blade_force = conditions["blade_force"]

        # 根據設計類型計算修正
        if self.params.design_type == InverseDesignType.EXIT_ANGLE:
            # 基於出口角度的設計
            result.compatible_exit_angle = self.params.target_exit_angle
            result.rotation_angle = self.compute_rotation_angle(
                conditions["exit_angle"], self.params.target_exit_angle
            )

        elif self.params.design_type == InverseDesignType.BLADE_FORCE:
            # 基於葉片力的設計
            result.compatible_exit_angle = self.compute_compatible_angle(
                conditions, self.params.target_blade_force, omega
            )
            result.rotation_angle = self.compute_rotation_angle(
                conditions["exit_angle"], result.compatible_exit_angle
            )

        elif self.params.design_type == InverseDesignType.PRESSURE_LOADING:
            # 基於壓力載荷的設計
            if target_pressure_ps is not None and target_pressure_ss is not None:
                # 計算目標葉片力
                target_force = self._compute_target_force(
                    target_pressure_ps, target_pressure_ss
                )
                result.compatible_exit_angle = self.compute_compatible_angle(
                    conditions, target_force, omega
                )
                result.rotation_angle = self.compute_rotation_angle(
                    conditions["exit_angle"], result.compatible_exit_angle
                )

        # 計算相容葉片數
        if abs(result.current_blade_force) > 1e-10:
            result.compatible_blade_number = int(
                n_blades
                * abs(self.params.target_blade_force / result.current_blade_force)
            )

        # 檢查收斂
        angle_error = abs(result.current_exit_angle - result.compatible_exit_angle)
        result.converged = angle_error < self.params.convergence_tolerance

        result.iterations = len(self.history) + 1
        self.history.append(result)

        return result

    def _compute_target_force(
        self,
        pressure_ps: NDArray[np.float64],
        pressure_ss: NDArray[np.float64],
    ) -> float:
        """計算目標葉片力。

        Args:
            pressure_ps: 壓力面壓力分佈
            pressure_ss: 吸力面壓力分佈

        Returns:
            目標葉片切向力
        """
        # 簡化：假設均勻面積
        dp = pressure_ps - pressure_ss
        return float(np.sum(dp))

    def apply_geometry_modification(
        self,
        rt_upper: NDArray[np.float64],
        rt_thickness: NDArray[np.float64],
        rotation_angle: float,
        r_surface: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """應用幾何修正。

        Args:
            rt_upper: 上表面 R*θ 座標
            rt_thickness: R*θ 厚度
            rotation_angle: 旋轉角度 [rad]
            r_surface: 半徑座標

        Returns:
            修正後的 (rt_upper, rt_thickness) 元組
        """
        # 應用旋轉
        rt_upper_new = rt_upper + rotation_angle * r_surface

        return rt_upper_new, rt_thickness.copy()

    def reset_history(self) -> None:
        """重置設計歷史。"""
        self.history = []

    def get_convergence_history(self) -> dict[str, list[float]]:
        """獲取收斂歷史。

        Returns:
            包含收斂歷史的字典
        """
        return {
            "iterations": list(range(1, len(self.history) + 1)),
            "exit_angle": [r.current_exit_angle for r in self.history],
            "blade_force": [r.current_blade_force for r in self.history],
            "rotation_angle": [r.rotation_angle for r in self.history],
        }
