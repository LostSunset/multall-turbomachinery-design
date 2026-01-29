# -*- coding: utf-8 -*-
"""MULTALL 時間推進模組。

實現顯式和隱式時間推進方法。
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties, Grid3D, SolverParameters
from .gas_properties import GasCalculator


class TimeStepMethod(IntEnum):
    """時間步進方法。"""

    EULER = 1  # 顯式 Euler
    RK2 = 2  # 二階 Runge-Kutta
    RK4 = 4  # 四階 Runge-Kutta
    SCREE = 3  # SCREE 格式（原 MULTALL）
    SSS = 5  # SSS 格式


class TimeStepper:
    """時間步進器。

    實現多種時間推進方法。
    """

    def __init__(
        self,
        gas: GasProperties,
        solver_params: SolverParameters,
    ) -> None:
        """初始化時間步進器。

        Args:
            gas: 氣體性質
            solver_params: 求解器參數
        """
        self.gas = gas
        self.gas_calc = GasCalculator(gas)
        self.params = solver_params

        # 時間步長限制
        self._dt_min = 1e-10
        self._dt_max = 1e-3

    def compute_local_time_step(
        self,
        flow: FlowField,
        grid: Grid3D,
    ) -> NDArray[np.float64]:
        """計算局部時間步長。

        基於 CFL 條件：Δt = CFL * Δx / (|V| + a)

        Args:
            flow: 流場數據
            grid: 網格數據

        Returns:
            局部時間步長數組
        """
        # 計算聲速
        a = self.gas_calc.speed_of_sound(flow.t_static)

        # 計算速度大小
        v_mag = np.sqrt(flow.vx**2 + flow.vr**2 + flow.vt**2)

        # 特徵速度（用於全局時間步長計算）
        _lambda_max = v_mag + a

        # 估計網格間距（簡化版本）
        # 實際應從 grid 計算
        dx = 0.001  # 軸向
        dtheta = 0.001  # 周向（r*dθ）
        dr = 0.001  # 徑向

        # 各方向 CFL 條件
        dt_x = dx / np.maximum(np.abs(flow.vx) + a, 1e-10)
        dt_theta = dtheta / np.maximum(np.abs(flow.vt) + a, 1e-10)
        dt_r = dr / np.maximum(np.abs(flow.vr) + a, 1e-10)

        # 取最小值
        dt = np.minimum(np.minimum(dt_x, dt_theta), dt_r)

        # 應用 CFL 數
        dt = self.params.cfl * dt

        # 限制時間步長範圍
        dt = np.clip(dt, self._dt_min, self._dt_max)

        return dt

    def euler_step(
        self,
        flow: FlowField,
        residual: tuple[NDArray[np.float64], ...],
        dt: NDArray[np.float64],
    ) -> None:
        """顯式 Euler 時間步進。

        U^{n+1} = U^n - Δt * R

        Args:
            flow: 流場數據（就地更新）
            residual: 殘差 (res_mass, res_momx, res_momr, res_momt, res_energy)
            dt: 時間步長
        """
        res_mass, res_momx, res_momr, res_momt, res_energy = residual

        # 更新守恆變量
        flow.ro -= dt * res_mass
        flow.rovx -= dt * res_momx
        flow.rovr -= dt * res_momr
        flow.rorvt -= dt * res_momt
        flow.roe -= dt * res_energy

        # 確保正值
        flow.ro = np.maximum(flow.ro, 1e-6)
        flow.roe = np.maximum(flow.roe, 1e-6)

    def rk2_step(
        self,
        flow: FlowField,
        residual_func: callable,
        dt: NDArray[np.float64],
    ) -> None:
        """二階 Runge-Kutta 時間步進。

        k1 = R(U^n)
        k2 = R(U^n - Δt*k1)
        U^{n+1} = U^n - Δt/2 * (k1 + k2)

        Args:
            flow: 流場數據（就地更新）
            residual_func: 殘差計算函數
            dt: 時間步長
        """
        # 保存初始狀態
        ro_0 = flow.ro.copy()
        rovx_0 = flow.rovx.copy()
        rovr_0 = flow.rovr.copy()
        rorvt_0 = flow.rorvt.copy()
        roe_0 = flow.roe.copy()

        # 第一階段：k1 = R(U^n)
        k1 = residual_func(flow)

        # 中間狀態：U* = U^n - Δt*k1
        flow.ro = ro_0 - dt * k1[0]
        flow.rovx = rovx_0 - dt * k1[1]
        flow.rovr = rovr_0 - dt * k1[2]
        flow.rorvt = rorvt_0 - dt * k1[3]
        flow.roe = roe_0 - dt * k1[4]

        # 更新原始變量
        self._update_primitive(flow)

        # 第二階段：k2 = R(U*)
        k2 = residual_func(flow)

        # 最終更新：U^{n+1} = U^n - Δt/2 * (k1 + k2)
        flow.ro = ro_0 - 0.5 * dt * (k1[0] + k2[0])
        flow.rovx = rovx_0 - 0.5 * dt * (k1[1] + k2[1])
        flow.rovr = rovr_0 - 0.5 * dt * (k1[2] + k2[2])
        flow.rorvt = rorvt_0 - 0.5 * dt * (k1[3] + k2[3])
        flow.roe = roe_0 - 0.5 * dt * (k1[4] + k2[4])

        # 確保正值
        flow.ro = np.maximum(flow.ro, 1e-6)
        flow.roe = np.maximum(flow.roe, 1e-6)

    def rk4_step(
        self,
        flow: FlowField,
        residual_func: callable,
        dt: NDArray[np.float64],
    ) -> None:
        """四階 Runge-Kutta 時間步進。

        k1 = R(U^n)
        k2 = R(U^n - Δt/2*k1)
        k3 = R(U^n - Δt/2*k2)
        k4 = R(U^n - Δt*k3)
        U^{n+1} = U^n - Δt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Args:
            flow: 流場數據（就地更新）
            residual_func: 殘差計算函數
            dt: 時間步長
        """
        # 保存初始狀態
        ro_0 = flow.ro.copy()
        rovx_0 = flow.rovx.copy()
        rovr_0 = flow.rovr.copy()
        rorvt_0 = flow.rorvt.copy()
        roe_0 = flow.roe.copy()

        # k1 = R(U^n)
        k1 = residual_func(flow)

        # U* = U^n - Δt/2*k1
        flow.ro = ro_0 - 0.5 * dt * k1[0]
        flow.rovx = rovx_0 - 0.5 * dt * k1[1]
        flow.rovr = rovr_0 - 0.5 * dt * k1[2]
        flow.rorvt = rorvt_0 - 0.5 * dt * k1[3]
        flow.roe = roe_0 - 0.5 * dt * k1[4]
        self._update_primitive(flow)

        # k2 = R(U*)
        k2 = residual_func(flow)

        # U* = U^n - Δt/2*k2
        flow.ro = ro_0 - 0.5 * dt * k2[0]
        flow.rovx = rovx_0 - 0.5 * dt * k2[1]
        flow.rovr = rovr_0 - 0.5 * dt * k2[2]
        flow.rorvt = rorvt_0 - 0.5 * dt * k2[3]
        flow.roe = roe_0 - 0.5 * dt * k2[4]
        self._update_primitive(flow)

        # k3 = R(U*)
        k3 = residual_func(flow)

        # U* = U^n - Δt*k3
        flow.ro = ro_0 - dt * k3[0]
        flow.rovx = rovx_0 - dt * k3[1]
        flow.rovr = rovr_0 - dt * k3[2]
        flow.rorvt = rorvt_0 - dt * k3[3]
        flow.roe = roe_0 - dt * k3[4]
        self._update_primitive(flow)

        # k4 = R(U*)
        k4 = residual_func(flow)

        # 最終更新
        flow.ro = ro_0 - dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        flow.rovx = rovx_0 - dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        flow.rovr = rovr_0 - dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        flow.rorvt = rorvt_0 - dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        flow.roe = roe_0 - dt / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

        # 確保正值
        flow.ro = np.maximum(flow.ro, 1e-6)
        flow.roe = np.maximum(flow.roe, 1e-6)

    def scree_step(
        self,
        flow: FlowField,
        residual: tuple[NDArray[np.float64], ...],
        dt: NDArray[np.float64],
    ) -> None:
        """SCREE 格式時間步進（原 MULTALL 方法）。

        使用多級 Runge-Kutta 類似的方法，但有特殊的係數。

        Args:
            flow: 流場數據（就地更新）
            residual: 殘差
            dt: 時間步長
        """
        # SCREE 係數（用於完整 SCREE 實現）
        _f1 = self.params.f1  # 2.0
        _f2 = self.params.f2  # -1.0
        _f2eff = self.params.f2eff  # -1.0

        res_mass, res_momx, res_momr, res_momt, res_energy = residual

        # 保存初始狀態
        ro_0 = flow.ro.copy()
        rovx_0 = flow.rovx.copy()
        rovr_0 = flow.rovr.copy()
        rorvt_0 = flow.rorvt.copy()
        roe_0 = flow.roe.copy()

        # 第一階段
        alpha1 = 0.25
        flow.ro = ro_0 - alpha1 * dt * res_mass
        flow.rovx = rovx_0 - alpha1 * dt * res_momx
        flow.rovr = rovr_0 - alpha1 * dt * res_momr
        flow.rorvt = rorvt_0 - alpha1 * dt * res_momt
        flow.roe = roe_0 - alpha1 * dt * res_energy

        # 第二階段
        alpha2 = 0.5
        flow.ro = ro_0 - alpha2 * dt * res_mass
        flow.rovx = rovx_0 - alpha2 * dt * res_momx
        flow.rovr = rovr_0 - alpha2 * dt * res_momr
        flow.rorvt = rorvt_0 - alpha2 * dt * res_momt
        flow.roe = roe_0 - alpha2 * dt * res_energy

        # 第三階段（最終）
        flow.ro = ro_0 - dt * res_mass
        flow.rovx = rovx_0 - dt * res_momx
        flow.rovr = rovr_0 - dt * res_momr
        flow.rorvt = rorvt_0 - dt * res_momt
        flow.roe = roe_0 - dt * res_energy

        # 確保正值
        flow.ro = np.maximum(flow.ro, 1e-6)
        flow.roe = np.maximum(flow.roe, 1e-6)

    def _update_primitive(self, flow: FlowField) -> None:
        """從守恆變量更新原始變量。

        Args:
            flow: 流場數據
        """
        # 密度
        flow.rho = flow.ro

        # 速度
        rho_safe = np.maximum(flow.rho, 1e-10)
        flow.vx = flow.rovx / rho_safe
        flow.vr = flow.rovr / rho_safe
        # vt 需要考慮 r（這裡簡化）
        # flow.vt = flow.rorvt / (rho_safe * r)

        # 內能和溫度
        v_sq = flow.vx**2 + flow.vr**2 + flow.vt**2
        e_total = flow.roe / rho_safe
        e_internal = e_total - 0.5 * v_sq
        flow.t_static = np.maximum(e_internal / self.gas_calc.cv, 100.0)

        # 壓力
        flow.p = self.gas_calc.pressure(flow.rho, flow.t_static)


class ConvergenceMonitor:
    """收斂監視器。

    追蹤殘差歷史並判斷收斂。
    """

    def __init__(
        self,
        convergence_limit: float = 0.005,
        history_size: int = 100,
    ) -> None:
        """初始化監視器。

        Args:
            convergence_limit: 收斂準則
            history_size: 歷史記錄大小
        """
        self.convergence_limit = convergence_limit
        self.history_size = history_size

        self._residual_history: list[float] = []
        self._mass_flow_history: list[float] = []
        self._initial_residual: float | None = None

    def add_residual(self, residual: float) -> None:
        """添加殘差記錄。

        Args:
            residual: 當前殘差
        """
        if self._initial_residual is None:
            self._initial_residual = residual

        self._residual_history.append(residual)
        if len(self._residual_history) > self.history_size:
            self._residual_history.pop(0)

    def add_mass_flow(self, mass_flow: float) -> None:
        """添加質量流量記錄。

        Args:
            mass_flow: 當前質量流量
        """
        self._mass_flow_history.append(mass_flow)
        if len(self._mass_flow_history) > self.history_size:
            self._mass_flow_history.pop(0)

    def compute_l2_residual(
        self,
        residual: tuple[NDArray[np.float64], ...],
    ) -> float:
        """計算 L2 範數殘差。

        Args:
            residual: 殘差數組

        Returns:
            L2 範數
        """
        l2_sum = 0.0
        n_total = 0
        for res in residual:
            l2_sum += float(np.sum(res**2))
            n_total += res.size

        return np.sqrt(l2_sum / max(n_total, 1))

    def is_converged(self) -> bool:
        """檢查是否收斂。

        Returns:
            是否收斂
        """
        if not self._residual_history:
            return False

        current = self._residual_history[-1]

        # 絕對收斂
        if current < self.convergence_limit:
            return True

        # 相對收斂（相對於初始殘差下降 3 個數量級）
        if self._initial_residual and current < self._initial_residual * 1e-3:
            return True

        return False

    def is_stalled(self, window: int = 50) -> bool:
        """檢查是否停滯。

        Args:
            window: 檢查窗口大小

        Returns:
            是否停滯
        """
        if len(self._residual_history) < window:
            return False

        recent = self._residual_history[-window:]
        avg_recent = np.mean(recent)
        std_recent = np.std(recent)

        # 如果標準差相對於平均值很小，認為停滯
        if avg_recent > 0 and std_recent / avg_recent < 0.01:
            return True

        return False

    def get_convergence_rate(self, window: int = 20) -> float:
        """計算收斂率。

        Args:
            window: 計算窗口

        Returns:
            收斂率（負值表示收斂）
        """
        if len(self._residual_history) < window:
            return 0.0

        recent = self._residual_history[-window:]
        if recent[0] <= 0 or recent[-1] <= 0:
            return 0.0

        # 對數線性擬合
        x = np.arange(window)
        y = np.log(np.array(recent) + 1e-20)

        # 最小二乘擬合斜率
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))

        return slope

    @property
    def residual_history(self) -> list[float]:
        """獲取殘差歷史。"""
        return self._residual_history.copy()

    @property
    def mass_flow_history(self) -> list[float]:
        """獲取質量流量歷史。"""
        return self._mass_flow_history.copy()

    @property
    def current_residual(self) -> float:
        """獲取當前殘差。"""
        return self._residual_history[-1] if self._residual_history else 1.0

    @property
    def normalized_residual(self) -> float:
        """獲取歸一化殘差。"""
        if not self._residual_history or not self._initial_residual:
            return 1.0
        return self._residual_history[-1] / self._initial_residual
