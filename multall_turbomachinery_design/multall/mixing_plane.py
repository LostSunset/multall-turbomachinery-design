# -*- coding: utf-8 -*-
"""MULTALL 混合平面模型模組。

實現用於多級渦輪機械的混合平面交界面處理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .data_structures import FlowField, GasProperties, MixingPlaneParameters
from .gas_properties import GasCalculator


class MixingPlaneType(IntEnum):
    """混合平面類型。"""

    CIRCUMFERENTIAL_AVERAGE = 1  # 周向平均
    FLUX_AVERAGE = 2  # 通量平均
    AREA_AVERAGE = 3  # 面積平均
    MASS_AVERAGE = 4  # 質量平均


@dataclass
class MixingPlaneInterface:
    """混合平面交界面數據。

    存儲交界面處的流場信息。
    """

    # 交界面位置
    j_upstream: int = 0  # 上游葉片排出口 J 索引
    j_downstream: int = 0  # 下游葉片排入口 J 索引

    # 周向平均後的流場量（沿 K 方向分布）
    rho_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    vx_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    vr_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    vt_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    p_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    t_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 總條件
    po_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    to_avg: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 質量流量分布
    mass_flux: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


class MixingPlaneModel:
    """混合平面模型。

    處理不同葉片排之間的交界面，通過周向平均傳遞流場信息。
    支持轉子-靜子和靜子-轉子交界面。
    """

    def __init__(
        self,
        gas: GasProperties,
        params: MixingPlaneParameters,
    ) -> None:
        """初始化混合平面模型。

        Args:
            gas: 氣體性質
            params: 混合平面參數
        """
        self.gas = gas
        self.gas_calc = GasCalculator(gas)
        self.params = params

        # 交界面列表
        self.interfaces: list[MixingPlaneInterface] = []

        # 平均類型
        self.averaging_type = MixingPlaneType.CIRCUMFERENTIAL_AVERAGE

    def add_interface(
        self,
        j_upstream: int,
        j_downstream: int,
    ) -> MixingPlaneInterface:
        """添加混合平面交界面。

        Args:
            j_upstream: 上游葉片排出口 J 索引
            j_downstream: 下游葉片排入口 J 索引

        Returns:
            創建的交界面
        """
        interface = MixingPlaneInterface(
            j_upstream=j_upstream,
            j_downstream=j_downstream,
        )
        self.interfaces.append(interface)
        return interface

    def compute_circumferential_average(
        self,
        flow: FlowField,
        j_index: int,
    ) -> dict[str, NDArray[np.float64]]:
        """計算周向平均值。

        對給定 J 索引處的流場進行周向（I 方向）平均。

        Args:
            flow: 流場數據
            j_index: J 索引

        Returns:
            周向平均後的流場量（沿 K 方向分布）
        """
        # 提取該截面的數據
        rho = flow.rho[:, j_index, :]  # (IM, KM)
        vx = flow.vx[:, j_index, :]
        vr = flow.vr[:, j_index, :]
        vt = flow.vt[:, j_index, :]
        p = flow.p[:, j_index, :]
        t = flow.t_static[:, j_index, :]

        # 周向平均（沿 I 方向）
        rho_avg = np.mean(rho, axis=0)  # (KM,)
        vx_avg = np.mean(vx, axis=0)
        vr_avg = np.mean(vr, axis=0)
        vt_avg = np.mean(vt, axis=0)
        p_avg = np.mean(p, axis=0)
        t_avg = np.mean(t, axis=0)

        # 計算總條件
        v_sq = vx_avg**2 + vr_avg**2 + vt_avg**2
        to_avg = t_avg + v_sq / (2 * self.gas.cp)
        po_avg = p_avg * (to_avg / t_avg) ** (self.gas.gamma / (self.gas.gamma - 1))

        return {
            "rho": rho_avg,
            "vx": vx_avg,
            "vr": vr_avg,
            "vt": vt_avg,
            "p": p_avg,
            "t": t_avg,
            "po": po_avg,
            "to": to_avg,
        }

    def compute_mass_average(
        self,
        flow: FlowField,
        j_index: int,
    ) -> dict[str, NDArray[np.float64]]:
        """計算質量平均值。

        對給定 J 索引處的流場進行質量加權平均。

        Args:
            flow: 流場數據
            j_index: J 索引

        Returns:
            質量平均後的流場量（沿 K 方向分布）
        """
        # 提取該截面的數據
        rho = flow.rho[:, j_index, :]
        vx = flow.vx[:, j_index, :]
        vr = flow.vr[:, j_index, :]
        vt = flow.vt[:, j_index, :]
        p = flow.p[:, j_index, :]
        t = flow.t_static[:, j_index, :]

        # 質量通量作為權重
        mass_flux = rho * vx  # (IM, KM)
        mass_flux_sum = np.sum(mass_flux, axis=0)  # (KM,)
        mass_flux_sum = np.maximum(mass_flux_sum, 1e-10)  # 避免除零

        # 質量平均
        rho_avg = np.mean(rho, axis=0)
        vx_avg = np.sum(mass_flux * vx, axis=0) / mass_flux_sum
        vr_avg = np.sum(mass_flux * vr, axis=0) / mass_flux_sum
        vt_avg = np.sum(mass_flux * vt, axis=0) / mass_flux_sum

        # 總焓質量平均
        ho = self.gas.cp * t + 0.5 * (vx**2 + vr**2 + vt**2)
        ho_avg = np.sum(mass_flux * ho, axis=0) / mass_flux_sum

        # 從總焓計算靜溫
        v_sq = vx_avg**2 + vr_avg**2 + vt_avg**2
        t_avg = (ho_avg - 0.5 * v_sq) / self.gas.cp

        # 壓力面積平均
        p_avg = np.mean(p, axis=0)

        # 密度從狀態方程
        rho_avg = p_avg / (self.gas.rgas * t_avg)

        # 計算總條件
        to_avg = t_avg + v_sq / (2 * self.gas.cp)
        po_avg = p_avg * (to_avg / t_avg) ** (self.gas.gamma / (self.gas.gamma - 1))

        return {
            "rho": rho_avg,
            "vx": vx_avg,
            "vr": vr_avg,
            "vt": vt_avg,
            "p": p_avg,
            "t": t_avg,
            "po": po_avg,
            "to": to_avg,
            "ho": ho_avg,
        }

    def compute_flux_average(
        self,
        flow: FlowField,
        j_index: int,
    ) -> dict[str, NDArray[np.float64]]:
        """計算通量平均值。

        對給定 J 索引處的流場進行守恆量通量平均。

        Args:
            flow: 流場數據
            j_index: J 索引

        Returns:
            通量平均後的流場量（沿 K 方向分布）
        """
        # 提取該截面的數據
        rho = flow.rho[:, j_index, :]
        vx = flow.vx[:, j_index, :]
        vr = flow.vr[:, j_index, :]
        vt = flow.vt[:, j_index, :]
        p = flow.p[:, j_index, :]
        t = flow.t_static[:, j_index, :]

        # 守恆量通量
        # 質量通量
        f_rho = rho * vx
        # 動量通量
        f_rhovx = rho * vx * vx + p
        f_rhovr = rho * vx * vr
        f_rhovt = rho * vx * vt
        # 能量通量
        e = self.gas.cp * t / self.gas.gamma + 0.5 * (vx**2 + vr**2 + vt**2)
        f_roe = rho * vx * (e + p / rho)

        # 周向平均通量
        f_rho_avg = np.mean(f_rho, axis=0)
        f_rhovx_avg = np.mean(f_rhovx, axis=0)
        f_rhovr_avg = np.mean(f_rhovr, axis=0)
        f_rhovt_avg = np.mean(f_rhovt, axis=0)
        f_roe_avg = np.mean(f_roe, axis=0)

        # 壓力面積平均
        p_avg = np.mean(p, axis=0)

        # 從通量重構原始變量
        # 這需要迭代求解，這裡使用簡化方法
        mass_flux_avg = f_rho_avg
        mass_flux_avg = np.maximum(mass_flux_avg, 1e-10)

        vx_avg = (f_rhovx_avg - p_avg) / mass_flux_avg
        vr_avg = f_rhovr_avg / mass_flux_avg
        vt_avg = f_rhovt_avg / mass_flux_avg

        # 從質量通量計算密度（需要假設 vx）
        vx_avg = np.maximum(vx_avg, 1.0)  # 確保正值
        rho_avg = mass_flux_avg / vx_avg

        # 從能量通量計算溫度
        e_avg = f_roe_avg / mass_flux_avg - p_avg / rho_avg
        v_sq = vx_avg**2 + vr_avg**2 + vt_avg**2
        t_avg = (e_avg - 0.5 * v_sq) * self.gas.gamma / self.gas.cp
        t_avg = np.maximum(t_avg, 200.0)  # 確保正值

        # 計算總條件
        to_avg = t_avg + v_sq / (2 * self.gas.cp)
        po_avg = p_avg * (to_avg / t_avg) ** (self.gas.gamma / (self.gas.gamma - 1))

        return {
            "rho": rho_avg,
            "vx": vx_avg,
            "vr": vr_avg,
            "vt": vt_avg,
            "p": p_avg,
            "t": t_avg,
            "po": po_avg,
            "to": to_avg,
        }

    def apply_mixing_plane(
        self,
        flow: FlowField,
        interface: MixingPlaneInterface,
    ) -> None:
        """應用混合平面邊界條件。

        將上游出口的周向平均值應用到下游入口。

        Args:
            flow: 流場數據（就地更新）
            interface: 交界面數據
        """
        # 根據平均類型計算上游平均值
        if self.averaging_type == MixingPlaneType.MASS_AVERAGE:
            avg = self.compute_mass_average(flow, interface.j_upstream)
        elif self.averaging_type == MixingPlaneType.FLUX_AVERAGE:
            avg = self.compute_flux_average(flow, interface.j_upstream)
        else:
            avg = self.compute_circumferential_average(flow, interface.j_upstream)

        # 更新交界面數據
        interface.rho_avg = avg["rho"]
        interface.vx_avg = avg["vx"]
        interface.vr_avg = avg["vr"]
        interface.vt_avg = avg["vt"]
        interface.p_avg = avg["p"]
        interface.t_avg = avg["t"]
        interface.po_avg = avg["po"]
        interface.to_avg = avg["to"]

        # 應用到下游入口
        j_down = interface.j_downstream
        im = flow.im

        for i in range(im):
            # 使用相同的周向平均值填充整個周向
            flow.rho[i, j_down, :] = avg["rho"]
            flow.vx[i, j_down, :] = avg["vx"]
            flow.vr[i, j_down, :] = avg["vr"]
            flow.vt[i, j_down, :] = avg["vt"]
            flow.p[i, j_down, :] = avg["p"]
            flow.t_static[i, j_down, :] = avg["t"]

    def apply_rotor_stator_interface(
        self,
        flow: FlowField,
        interface: MixingPlaneInterface,
        omega_upstream: float,
        omega_downstream: float,
    ) -> None:
        """應用轉子-靜子交界面。

        在不同轉速的葉片排之間進行坐標轉換。

        Args:
            flow: 流場數據（就地更新）
            interface: 交界面數據
            omega_upstream: 上游葉片排角速度 [rad/s]
            omega_downstream: 下游葉片排角速度 [rad/s]
        """
        # 首先計算上游平均值
        avg = self.compute_circumferential_average(flow, interface.j_upstream)

        # 獲取半徑（假設從網格獲取，這裡使用默認值）
        km = len(avg["vt"])
        r = np.linspace(0.25, 0.35, km)  # 預設半徑分布

        # 在絕對坐標系中的周向速度
        vt_abs_upstream = avg["vt"]

        # 轉換到下游參考系
        # V_θ_abs = V_θ_rel + ω * r
        delta_omega = omega_downstream - omega_upstream
        vt_abs_downstream = vt_abs_upstream - delta_omega * r

        # 更新平均值
        avg["vt"] = vt_abs_downstream

        # 更新交界面數據
        interface.rho_avg = avg["rho"]
        interface.vx_avg = avg["vx"]
        interface.vr_avg = avg["vr"]
        interface.vt_avg = avg["vt"]
        interface.p_avg = avg["p"]
        interface.t_avg = avg["t"]
        interface.po_avg = avg["po"]
        interface.to_avg = avg["to"]

        # 應用到下游入口
        j_down = interface.j_downstream
        im = flow.im

        for i in range(im):
            flow.rho[i, j_down, :] = avg["rho"]
            flow.vx[i, j_down, :] = avg["vx"]
            flow.vr[i, j_down, :] = avg["vr"]
            flow.vt[i, j_down, :] = avg["vt"]
            flow.p[i, j_down, :] = avg["p"]
            flow.t_static[i, j_down, :] = avg["t"]

    def compute_interface_mass_flow(
        self,
        flow: FlowField,
        j_index: int,
    ) -> float:
        """計算交界面處的質量流量。

        Args:
            flow: 流場數據
            j_index: J 索引

        Returns:
            質量流量 [kg/s]
        """
        # 提取該截面的數據
        rho = flow.rho[:, j_index, :]
        vx = flow.vx[:, j_index, :]

        # 質量通量
        mass_flux = rho * vx

        # 需要乘以面積，這裡簡化
        # 假設均勻面積
        area_per_cell = 0.001  # m^2

        return float(np.sum(mass_flux) * area_per_cell)

    def compute_interface_efficiency(
        self,
        flow: FlowField,
        interface: MixingPlaneInterface,
    ) -> dict[str, float]:
        """計算交界面處的效率相關量。

        Args:
            flow: 流場數據
            interface: 交界面數據

        Returns:
            效率相關量字典
        """
        # 計算上下游平均值
        avg_up = self.compute_mass_average(flow, interface.j_upstream)
        avg_down = self.compute_mass_average(flow, interface.j_downstream)

        # 總壓損失
        po_loss = (avg_up["po"].mean() - avg_down["po"].mean()) / avg_up["po"].mean()

        # 總溫比
        to_ratio = avg_down["to"].mean() / avg_up["to"].mean()

        # 壓比
        p_ratio = avg_down["p"].mean() / avg_up["p"].mean()

        # 等熵效率（近似）
        gamma = self.gas.gamma
        eta_isen = (p_ratio ** ((gamma - 1) / gamma) - 1) / (to_ratio - 1 + 1e-10)
        eta_isen = np.clip(eta_isen, 0.0, 1.0)

        return {
            "total_pressure_loss": po_loss,
            "total_temperature_ratio": to_ratio,
            "pressure_ratio": p_ratio,
            "isentropic_efficiency": eta_isen,
        }

    def update_all_interfaces(self, flow: FlowField) -> None:
        """更新所有混合平面交界面。

        Args:
            flow: 流場數據（就地更新）
        """
        for interface in self.interfaces:
            self.apply_mixing_plane(flow, interface)


class NonReflectingBoundary:
    """無反射邊界條件。

    用於減少混合平面處的數值反射。
    """

    def __init__(
        self,
        gas: GasProperties,
        relaxation_factor: float = 0.1,
    ) -> None:
        """初始化無反射邊界。

        Args:
            gas: 氣體性質
            relaxation_factor: 鬆弛因子
        """
        self.gas = gas
        self.gas_calc = GasCalculator(gas)
        self.relaxation_factor = relaxation_factor

    def apply_inlet_nrbc(
        self,
        flow: FlowField,
        j_index: int,
        target_po: NDArray[np.float64],
        target_to: NDArray[np.float64],
        target_alpha: NDArray[np.float64],
    ) -> None:
        """應用入口無反射邊界條件。

        基於特徵變量的無反射邊界條件。

        Args:
            flow: 流場數據（就地更新）
            j_index: J 索引
            target_po: 目標總壓分布
            target_to: 目標總溫分布
            target_alpha: 目標流動角分布
        """
        im, km = flow.im, flow.km
        sigma = self.relaxation_factor

        for i in range(im):
            for k in range(km):
                # 當前狀態
                rho = flow.rho[i, j_index, k]
                vx = flow.vx[i, j_index, k]
                p = flow.p[i, j_index, k]
                t = flow.t_static[i, j_index, k]

                # 聲速
                a = self.gas_calc.speed_of_sound(t)

                # 目標狀態
                po_t = target_po[k]
                to_t = target_to[k]

                # 當前馬赫數
                v_mag = abs(vx)
                mach = v_mag / max(a, 1.0)
                mach = min(mach, 0.99)

                # 從目標總條件計算目標靜態條件
                p_t, t_t = self.gas_calc.static_from_total(po_t, to_t, mach)
                rho_t = self.gas_calc.density(p_t, t_t)

                # 鬆弛更新（僅更新入射特徵）
                # 對於亞音速入口，更新壓力和溫度
                flow.p[i, j_index, k] = p + sigma * (p_t - p)
                flow.t_static[i, j_index, k] = t + sigma * (t_t - t)
                flow.rho[i, j_index, k] = rho + sigma * (rho_t - rho)

    def apply_exit_nrbc(
        self,
        flow: FlowField,
        j_index: int,
        target_p: NDArray[np.float64],
    ) -> None:
        """應用出口無反射邊界條件。

        基於特徵變量的無反射邊界條件。

        Args:
            flow: 流場數據（就地更新）
            j_index: J 索引
            target_p: 目標靜壓分布
        """
        im, km = flow.im, flow.km
        sigma = self.relaxation_factor

        for i in range(im):
            for k in range(km):
                # 當前狀態
                p = flow.p[i, j_index, k]

                # 目標靜壓
                p_t = target_p[k]

                # 鬆弛更新（僅更新出射特徵）
                # 對於亞音速出口，只更新壓力
                flow.p[i, j_index, k] = p + sigma * (p_t - p)
