# -*- coding: utf-8 -*-
"""MULTALL 後處理工具模組。

提供流場結果後處理功能：
- 性能指標計算（效率、壓比、功率）
- 流場可視化數據提取
- 損失分解分析
- 結果導出（VTK、CSV 等格式）

基於 FORTRAN OUTPUT 子程序移植。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties, Grid3D


@dataclass
class PerformanceMetrics:
    """性能指標。"""

    # 整機性能
    mass_flow: float = 0.0  # 質量流量 [kg/s]
    power_output: float = 0.0  # 功率輸出 [kW]
    total_to_total_efficiency: float = 0.0  # 總-總效率
    total_to_static_efficiency: float = 0.0  # 總-靜效率
    pressure_ratio: float = 0.0  # 總壓比
    temperature_ratio: float = 0.0  # 總溫比

    # 入口條件
    inlet_total_pressure: float = 0.0  # 入口總壓 [Pa]
    inlet_total_temperature: float = 0.0  # 入口總溫 [K]
    inlet_mach: float = 0.0  # 入口馬赫數

    # 出口條件
    exit_total_pressure: float = 0.0  # 出口總壓 [Pa]
    exit_static_pressure: float = 0.0  # 出口靜壓 [Pa]
    exit_total_temperature: float = 0.0  # 出口總溫 [K]
    exit_mach: float = 0.0  # 出口馬赫數

    # 損失係數
    total_pressure_loss_coefficient: float = 0.0  # 總壓損失係數
    entropy_increase: float = 0.0  # 熵增 [J/(kg·K)]


@dataclass
class StagePerformance:
    """級性能。"""

    stage_number: int = 1

    # 級性能指標
    pressure_ratio: float = 0.0
    temperature_ratio: float = 0.0
    isentropic_efficiency: float = 0.0
    reaction: float = 0.0  # 反動度
    work_coefficient: float = 0.0  # 負荷係數
    flow_coefficient: float = 0.0  # 流量係數

    # 損失分解
    profile_loss: float = 0.0  # 葉型損失
    secondary_loss: float = 0.0  # 二次流損失
    tip_clearance_loss: float = 0.0  # 葉尖間隙損失
    annulus_loss: float = 0.0  # 環形損失


@dataclass
class FlowVisualizationData:
    """流場可視化數據。"""

    # 網格座標
    x: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    r: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    theta: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 流場變量
    pressure: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    temperature: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    mach: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    mach_relative: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    entropy_function: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 速度分量
    vx: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    vr: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    vt: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


class PerformanceCalculator:
    """性能計算器。"""

    def __init__(self, gas: GasProperties):
        """初始化。

        Args:
            gas: 氣體性質
        """
        self.gas = gas

    def compute_overall_performance(
        self,
        flow: FlowField,
        grid: Grid3D | None = None,
        omega: float = 0.0,
    ) -> PerformanceMetrics:
        """計算整機性能。

        Args:
            flow: 流場
            grid: 網格（可選）
            omega: 角速度 [rad/s]

        Returns:
            性能指標
        """
        metrics = PerformanceMetrics()
        jm = flow.jm

        # 入口和出口索引
        j_in = 1
        j_out = jm - 2

        # 計算入口平均條件
        rho_in = np.mean(flow.rho[:, j_in, :])
        vx_in = np.mean(flow.vx[:, j_in, :])
        p_in = np.mean(flow.p[:, j_in, :])
        t_in = np.mean(flow.t_static[:, j_in, :])

        # 計算出口平均條件
        p_out = np.mean(flow.p[:, j_out, :])
        t_out = np.mean(flow.t_static[:, j_out, :])

        # 速度大小
        v_in = np.sqrt(
            np.mean(flow.vx[:, j_in, :] ** 2)
            + np.mean(flow.vr[:, j_in, :] ** 2)
            + np.mean(flow.vt[:, j_in, :] ** 2)
        )
        v_out = np.sqrt(
            np.mean(flow.vx[:, j_out, :] ** 2)
            + np.mean(flow.vr[:, j_out, :] ** 2)
            + np.mean(flow.vt[:, j_out, :] ** 2)
        )

        # 總溫
        cp = self.gas.cp
        gamma = self.gas.gamma
        to_in = t_in + v_in**2 / (2 * cp)
        to_out = t_out + v_out**2 / (2 * cp)

        # 總壓（等熵關係）
        po_in = p_in * (to_in / t_in) ** (gamma / (gamma - 1))
        po_out = p_out * (to_out / t_out) ** (gamma / (gamma - 1))

        # 馬赫數
        a_in = np.sqrt(gamma * self.gas.rgas * t_in)
        a_out = np.sqrt(gamma * self.gas.rgas * t_out)
        mach_in = v_in / a_in
        mach_out = v_out / a_out

        # 估算面積（簡化）
        area_in = 1.0  # 需要從網格計算

        # 質量流量
        mass_flow = rho_in * vx_in * area_in

        # 功率（焓降 × 質量流量）
        delta_ho = cp * (to_in - to_out)
        power = mass_flow * delta_ho / 1000.0  # kW

        # 壓比和溫比
        pr = po_in / po_out  # 渦輪膨脹比
        tr = to_in / to_out

        # 等熵效率
        # 總-總效率：η_tt = (1 - T_out/T_in) / (1 - (P_out/P_in)^((γ-1)/γ))
        exp = (gamma - 1) / gamma
        if abs(1 - (1 / pr) ** exp) > 1e-10:
            eta_tt = (1 - 1 / tr) / (1 - (1 / pr) ** exp)
        else:
            eta_tt = 0.0

        # 總-靜效率
        if abs(1 - (p_out / po_in) ** exp) > 1e-10:
            eta_ts = (1 - 1 / tr) / (1 - (p_out / po_in) ** exp)
        else:
            eta_ts = 0.0

        # 總壓損失係數
        loss_coeff = (po_in - po_out) / (po_in - p_in) if (po_in - p_in) > 0 else 0.0

        # 熵增
        r_gas = self.gas.rgas
        ds = cp * np.log(to_out / to_in) - r_gas * np.log(po_out / po_in)

        # 存儲結果
        metrics.mass_flow = mass_flow
        metrics.power_output = power
        metrics.total_to_total_efficiency = eta_tt
        metrics.total_to_static_efficiency = eta_ts
        metrics.pressure_ratio = pr
        metrics.temperature_ratio = tr
        metrics.inlet_total_pressure = po_in
        metrics.inlet_total_temperature = to_in
        metrics.inlet_mach = mach_in
        metrics.exit_total_pressure = po_out
        metrics.exit_static_pressure = p_out
        metrics.exit_total_temperature = to_out
        metrics.exit_mach = mach_out
        metrics.total_pressure_loss_coefficient = loss_coeff
        metrics.entropy_increase = ds

        return metrics

    def compute_stage_performance(
        self,
        flow: FlowField,
        j_inlet: int,
        j_exit: int,
        omega: float = 0.0,
        r_mean: float = 0.5,
    ) -> StagePerformance:
        """計算級性能。

        Args:
            flow: 流場
            j_inlet: 級入口 J 索引
            j_exit: 級出口 J 索引
            omega: 角速度 [rad/s]
            r_mean: 平均半徑 [m]

        Returns:
            級性能
        """
        perf = StagePerformance()

        cp = self.gas.cp
        gamma = self.gas.gamma

        # 入口條件
        vx_in = np.mean(flow.vx[:, j_inlet, :])
        vt_in = np.mean(flow.vt[:, j_inlet, :])
        p_in = np.mean(flow.p[:, j_inlet, :])
        t_in = np.mean(flow.t_static[:, j_inlet, :])

        # 出口條件
        vx_out = np.mean(flow.vx[:, j_exit, :])
        vt_out = np.mean(flow.vt[:, j_exit, :])
        p_out = np.mean(flow.p[:, j_exit, :])
        t_out = np.mean(flow.t_static[:, j_exit, :])

        # 軸向速度
        vm = 0.5 * (vx_in + vx_out)

        # 輪緣速度
        u = omega * r_mean

        # 總溫和總壓
        v_in = np.sqrt(vx_in**2 + vt_in**2)
        v_out = np.sqrt(vx_out**2 + vt_out**2)
        to_in = t_in + v_in**2 / (2 * cp)
        to_out = t_out + v_out**2 / (2 * cp)
        po_in = p_in * (to_in / t_in) ** (gamma / (gamma - 1))
        po_out = p_out * (to_out / t_out) ** (gamma / (gamma - 1))

        # 壓比和溫比
        perf.pressure_ratio = po_in / po_out
        perf.temperature_ratio = to_in / to_out

        # 負荷係數 ψ = Δh / U²
        delta_ho = cp * (to_in - to_out)
        if u > 1e-10:
            perf.work_coefficient = delta_ho / (u * u)
        else:
            perf.work_coefficient = 0.0

        # 流量係數 φ = Vm / U
        if u > 1e-10:
            perf.flow_coefficient = vm / u
        else:
            perf.flow_coefficient = 0.0

        # 反動度 R = (h_stator - h_rotor) / Δh_stage
        # 簡化：R ≈ 1 - (Vt_in + Vt_out) / (2U)
        if u > 1e-10:
            perf.reaction = 1 - (vt_in + vt_out) / (2 * u)
        else:
            perf.reaction = 0.5

        # 等熵效率
        exp = (gamma - 1) / gamma
        if abs(1 - (1 / perf.pressure_ratio) ** exp) > 1e-10:
            perf.isentropic_efficiency = (1 - 1 / perf.temperature_ratio) / (
                1 - (1 / perf.pressure_ratio) ** exp
            )
        else:
            perf.isentropic_efficiency = 0.0

        return perf

    def compute_entropy_function(
        self, flow: FlowField, po_ref: float, to_ref: float
    ) -> NDArray[np.float64]:
        """計算熵函數。

        熵函數 S = p/p_ref * (T_ref/T)^(γ/(γ-1))

        Args:
            flow: 流場
            po_ref: 參考總壓 [Pa]
            to_ref: 參考總溫 [K]

        Returns:
            熵函數陣列
        """
        gamma = self.gas.gamma
        exp = gamma / (gamma - 1)

        # 使用靜溫
        t_static = flow.t_static.copy()

        # 確保溫度為正
        t_static = np.maximum(t_static, 1.0)

        # 計算熵函數
        entropy_func = flow.p / po_ref * (to_ref / t_static) ** exp

        return entropy_func


class FlowFieldExtractor:
    """流場數據提取器。"""

    def __init__(self, gas: GasProperties):
        """初始化。

        Args:
            gas: 氣體性質
        """
        self.gas = gas

    def extract_at_j_station(
        self, flow: FlowField, grid: Grid3D | None, j_index: int
    ) -> dict[str, NDArray[np.float64]]:
        """提取指定 J 站的流場數據。

        Args:
            flow: 流場
            grid: 網格（可選）
            j_index: J 索引

        Returns:
            包含流場變量的字典
        """
        data = {
            "rho": flow.rho[:, j_index, :].copy(),
            "vx": flow.vx[:, j_index, :].copy(),
            "vr": flow.vr[:, j_index, :].copy(),
            "vt": flow.vt[:, j_index, :].copy(),
            "p": flow.p[:, j_index, :].copy(),
            "t_static": flow.t_static[:, j_index, :].copy(),
        }

        # 計算派生量
        gamma = self.gas.gamma
        cp = self.gas.cp

        # 速度大小
        v_mag = np.sqrt(data["vx"] ** 2 + data["vr"] ** 2 + data["vt"] ** 2)
        data["velocity"] = v_mag

        # 馬赫數
        a = np.sqrt(gamma * self.gas.rgas * data["t_static"])
        data["mach"] = v_mag / a

        # 總溫和總壓
        data["t_total"] = data["t_static"] + v_mag**2 / (2 * cp)
        data["p_total"] = data["p"] * (data["t_total"] / data["t_static"]) ** (gamma / (gamma - 1))

        return data

    def extract_at_k_surface(
        self, flow: FlowField, grid: Grid3D | None, k_index: int
    ) -> dict[str, NDArray[np.float64]]:
        """提取指定 K 流線面的流場數據。

        Args:
            flow: 流場
            grid: 網格（可選）
            k_index: K 索引

        Returns:
            包含流場變量的字典
        """
        data = {
            "rho": flow.rho[:, :, k_index].copy(),
            "vx": flow.vx[:, :, k_index].copy(),
            "vr": flow.vr[:, :, k_index].copy(),
            "vt": flow.vt[:, :, k_index].copy(),
            "p": flow.p[:, :, k_index].copy(),
            "t_static": flow.t_static[:, :, k_index].copy(),
        }

        # 計算派生量
        gamma = self.gas.gamma
        cp = self.gas.cp

        v_mag = np.sqrt(data["vx"] ** 2 + data["vr"] ** 2 + data["vt"] ** 2)
        data["velocity"] = v_mag

        a = np.sqrt(gamma * self.gas.rgas * data["t_static"])
        data["mach"] = v_mag / a

        data["t_total"] = data["t_static"] + v_mag**2 / (2 * cp)
        data["p_total"] = data["p"] * (data["t_total"] / data["t_static"]) ** (gamma / (gamma - 1))

        return data

    def extract_blade_surface_data(
        self,
        flow: FlowField,
        i_ps: int = 0,
        i_ss: int = -1,
    ) -> dict[str, NDArray[np.float64]]:
        """提取葉片表面數據。

        Args:
            flow: 流場
            i_ps: 壓力面 I 索引
            i_ss: 吸力面 I 索引

        Returns:
            包含表面壓力和速度的字典
        """
        data = {
            "p_ps": flow.p[i_ps, :, :].copy(),  # 壓力面壓力
            "p_ss": flow.p[i_ss, :, :].copy(),  # 吸力面壓力
            "vx_ps": flow.vx[i_ps, :, :].copy(),
            "vx_ss": flow.vx[i_ss, :, :].copy(),
            "vt_ps": flow.vt[i_ps, :, :].copy(),
            "vt_ss": flow.vt[i_ss, :, :].copy(),
        }

        # 計算壓力係數
        p_avg = 0.5 * (np.mean(data["p_ps"]) + np.mean(data["p_ss"]))
        v_avg = 0.5 * (
            np.mean(np.sqrt(data["vx_ps"] ** 2 + data["vt_ps"] ** 2))
            + np.mean(np.sqrt(data["vx_ss"] ** 2 + data["vt_ss"] ** 2))
        )
        rho_avg = np.mean(flow.rho)
        q_ref = 0.5 * rho_avg * v_avg**2

        if q_ref > 0:
            data["cp_ps"] = (data["p_ps"] - p_avg) / q_ref
            data["cp_ss"] = (data["p_ss"] - p_avg) / q_ref
        else:
            data["cp_ps"] = np.zeros_like(data["p_ps"])
            data["cp_ss"] = np.zeros_like(data["p_ss"])

        return data

    def create_visualization_data(
        self, flow: FlowField, grid: Grid3D | None = None
    ) -> FlowVisualizationData:
        """創建可視化數據。

        Args:
            flow: 流場
            grid: 網格（可選）

        Returns:
            可視化數據
        """
        viz = FlowVisualizationData()

        gamma = self.gas.gamma

        # 流場變量
        viz.pressure = flow.p.copy()
        viz.temperature = flow.t_static.copy()
        viz.vx = flow.vx.copy()
        viz.vr = flow.vr.copy()
        viz.vt = flow.vt.copy()

        # 速度大小和馬赫數
        v_mag = np.sqrt(flow.vx**2 + flow.vr**2 + flow.vt**2)
        a = np.sqrt(gamma * self.gas.rgas * flow.t_static)
        viz.mach = v_mag / a

        # 網格座標
        if grid is not None:
            viz.x = grid.x.copy()
            viz.r = grid.r.copy()
            viz.theta = grid.theta.copy()

        return viz


class ResultExporter:
    """結果導出器。"""

    def __init__(self, gas: GasProperties):
        """初始化。

        Args:
            gas: 氣體性質
        """
        self.gas = gas
        self.calculator = PerformanceCalculator(gas)
        self.extractor = FlowFieldExtractor(gas)

    def export_performance_summary(
        self,
        metrics: PerformanceMetrics,
        output_path: Path | str,
    ) -> None:
        """導出性能摘要。

        Args:
            metrics: 性能指標
            output_path: 輸出路徑
        """
        output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("         MULTALL 性能計算結果\n")
            f.write("=" * 60 + "\n\n")

            f.write("整機性能\n")
            f.write("-" * 40 + "\n")
            f.write(f"  質量流量:              {metrics.mass_flow:.4f} kg/s\n")
            f.write(f"  功率輸出:              {metrics.power_output:.2f} kW\n")
            f.write(f"  總-總效率:             {metrics.total_to_total_efficiency:.4f}\n")
            f.write(f"  總-靜效率:             {metrics.total_to_static_efficiency:.4f}\n")
            f.write(f"  壓比:                  {metrics.pressure_ratio:.4f}\n")
            f.write(f"  溫比:                  {metrics.temperature_ratio:.4f}\n\n")

            f.write("入口條件\n")
            f.write("-" * 40 + "\n")
            f.write(f"  總壓:                  {metrics.inlet_total_pressure:.0f} Pa\n")
            f.write(f"  總溫:                  {metrics.inlet_total_temperature:.2f} K\n")
            f.write(f"  馬赫數:                {metrics.inlet_mach:.4f}\n\n")

            f.write("出口條件\n")
            f.write("-" * 40 + "\n")
            f.write(f"  總壓:                  {metrics.exit_total_pressure:.0f} Pa\n")
            f.write(f"  靜壓:                  {metrics.exit_static_pressure:.0f} Pa\n")
            f.write(f"  總溫:                  {metrics.exit_total_temperature:.2f} K\n")
            f.write(f"  馬赫數:                {metrics.exit_mach:.4f}\n\n")

            f.write("損失分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  總壓損失係數:          {metrics.total_pressure_loss_coefficient:.4f}\n")
            f.write(f"  熵增:                  {metrics.entropy_increase:.4f} J/(kg·K)\n")

    def export_station_data_csv(
        self,
        data: dict[str, NDArray[np.float64]],
        output_path: Path | str,
    ) -> None:
        """導出站點數據為 CSV。

        Args:
            data: 站點數據字典
            output_path: 輸出路徑
        """
        output_path = Path(output_path)

        # 獲取數據形狀
        first_key = next(iter(data.keys()))
        shape = data[first_key].shape

        with open(output_path, "w", encoding="utf-8") as f:
            # 寫入表頭
            headers = ["i", "k"] + list(data.keys())
            f.write(",".join(headers) + "\n")

            # 寫入數據
            for i in range(shape[0]):
                for k in range(shape[1]):
                    row = [str(i), str(k)]
                    for key in data.keys():
                        row.append(f"{data[key][i, k]:.6e}")
                    f.write(",".join(row) + "\n")

    def export_flow_field_binary(
        self,
        flow: FlowField,
        output_path: Path | str,
    ) -> None:
        """導出流場為二進制格式。

        Args:
            flow: 流場
            output_path: 輸出路徑
        """
        output_path = Path(output_path)

        with open(output_path, "wb") as f:
            # 寫入網格尺寸
            np.array([flow.im, flow.jm, flow.km], dtype=np.int32).tofile(f)

            # 寫入守恆變量
            flow.rho.astype(np.float64).tofile(f)
            flow.vx.astype(np.float64).tofile(f)
            flow.vr.astype(np.float64).tofile(f)
            flow.vt.astype(np.float64).tofile(f)
            flow.p.astype(np.float64).tofile(f)

    def export_convergence_history(
        self,
        residuals: list[float],
        output_path: Path | str,
    ) -> None:
        """導出收斂歷史。

        Args:
            residuals: 殘差列表
            output_path: 輸出路徑
        """
        output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("iteration,residual\n")
            for i, res in enumerate(residuals):
                f.write(f"{i + 1},{res:.6e}\n")
