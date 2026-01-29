# -*- coding: utf-8 -*-
"""流表面生成模組。

提供軸向流和混流的流表面生成、平滑和操作功能。
"""

from __future__ import annotations

import math

import numpy as np

from .data_structures import CONSTANTS, FlowType, StreamSurface


class StreamSurfaceGenerator:
    """流表面生成器。"""

    def __init__(self, flow_type: FlowType) -> None:
        """初始化流表面生成器。

        Args:
            flow_type: 流動類型（軸向或混流）
        """
        self.flow_type = flow_type
        self.deg2rad = CONSTANTS["DEG2RAD"]
        self.rad2deg = CONSTANTS["RAD2DEG"]

    def generate_axial_surface(
        self,
        r_design: float,
        axial_chord_1: float,
        axial_chord_2: float,
        row_gap: float,
        stage_gap: float,
        npoints: int = 9,
    ) -> StreamSurface:
        """生成軸向流流表面。

        Args:
            r_design: 設計半徑 [m]
            axial_chord_1: 轉子軸向弦長 [m]
            axial_chord_2: 定子軸向弦長 [m]
            row_gap: 行間隙 [m]
            stage_gap: 級間隙 [m]
            npoints: 流表面點數

        Returns:
            流表面
        """
        # 生成標準 9 點流表面
        # 進1, 進0.5, 進0, TE1, gap, LE2, 出1, 出0.5, 出0
        x = []
        r = []

        # 轉子上游
        x.append(-axial_chord_1)  # 進1
        x.append(-0.5 * axial_chord_1)  # 進0.5
        x.append(0.0)  # 進0 (轉子前緣)

        # 轉子
        x.append(axial_chord_1)  # TE1 (轉子後緣)

        # 行間隙
        x.append(axial_chord_1 + 0.5 * row_gap)  # gap 中間

        # 定子
        x.append(axial_chord_1 + row_gap)  # LE2 (定子前緣)
        x.append(axial_chord_1 + row_gap + axial_chord_2)  # 定子後緣

        # 定子下游
        x.append(axial_chord_1 + row_gap + axial_chord_2 + 0.5 * stage_gap)  # 出0.5
        x.append(axial_chord_1 + row_gap + axial_chord_2 + stage_gap)  # 出0

        # 所有點的半徑相同（軸向流）
        r = [r_design] * len(x)

        # 子午速度比（假設為1）
        vm_ratio = [1.0] * len(x)

        # 計算流表面距離
        s_dist = self._calculate_distance(x, r)

        # 俯仰角（軸向流為0）
        pitch_angle = [0.0] * len(x)

        return StreamSurface(
            npoints=len(x),
            x=x,
            r=r,
            vm_ratio=vm_ratio,
            pitch_angle=pitch_angle,
            s_dist=s_dist,
            fblock=[0.0] * len(x),
        )

    def generate_mixed_surface(
        self,
        x_coords: list[float],
        r_coords: list[float],
        vm_ratios: list[float],
        nle1: int,
        nte1: int,
        nle2: int,
        nte2: int,
    ) -> StreamSurface:
        """生成混流流表面。

        Args:
            x_coords: 軸向坐標列表 [m]
            r_coords: 徑向坐標列表 [m]
            vm_ratios: 子午速度比列表
            nle1: 轉子前緣點索引
            nte1: 轉子後緣點索引
            nle2: 定子前緣點索引
            nte2: 定子後緣點索引

        Returns:
            流表面
        """
        npoints = len(x_coords)

        # 計算俯仰角
        pitch_angle = self._calculate_pitch_angle(x_coords, r_coords)

        # 計算流表面距離
        s_dist = self._calculate_distance(x_coords, r_coords)

        return StreamSurface(
            npoints=npoints,
            x=x_coords,
            r=r_coords,
            vm_ratio=vm_ratios,
            pitch_angle=pitch_angle,
            s_dist=s_dist,
            fblock=[0.0] * npoints,
        )

    def _calculate_distance(self, x: list[float], r: list[float]) -> list[float]:
        """計算沿流表面的累積距離。

        Args:
            x: 軸向坐標 [m]
            r: 徑向坐標 [m]

        Returns:
            累積距離 [m]
        """
        s_dist = [0.0]
        for i in range(1, len(x)):
            dx = x[i] - x[i - 1]
            dr = r[i] - r[i - 1]
            ds = math.sqrt(dx * dx + dr * dr)
            s_dist.append(s_dist[-1] + ds)
        return s_dist

    def _calculate_pitch_angle(self, x: list[float], r: list[float]) -> list[float]:
        """計算俯仰角。

        Args:
            x: 軸向坐標 [m]
            r: 徑向坐標 [m]

        Returns:
            俯仰角 [度]
        """
        pitch_angle = []
        n = len(x)

        for i in range(n):
            if i == 0:
                # 第一點：使用前向差分
                dx = x[1] - x[0]
                dr = r[1] - r[0]
            elif i == n - 1:
                # 最後一點：使用後向差分
                dx = x[i] - x[i - 1]
                dr = r[i] - r[i - 1]
            else:
                # 中間點：使用中心差分
                dx = x[i + 1] - x[i - 1]
                dr = r[i + 1] - r[i - 1]

            # 俯仰角 = arctan(dr/dx)
            angle = math.atan2(dr, dx) * self.rad2deg
            pitch_angle.append(angle)

        return pitch_angle

    def apply_blockage_factor(
        self,
        surface: StreamSurface,
        fblock_le: float,
        fblock_te: float,
        nle: int,
        nte: int,
    ) -> None:
        """應用堵塞因子到流表面。

        Args:
            surface: 流表面
            fblock_le: 前緣堵塞因子
            fblock_te: 後緣堵塞因子
            nle: 前緣點索引
            nte: 後緣點索引
        """
        # 在葉片區域線性插值堵塞因子
        for i in range(surface.npoints):
            if i < nle:
                # 上游區域：無堵塞
                surface.fblock[i] = 0.0
            elif i > nte:
                # 下游區域：保持後緣值
                surface.fblock[i] = fblock_te
            else:
                # 葉片區域：線性插值
                frac = (i - nle) / (nte - nle) if nte > nle else 0.0
                surface.fblock[i] = fblock_le + (fblock_te - fblock_le) * frac

    def smooth_surface(
        self, surface: StreamSurface, nsmooth: int = 5, smooth_factor: float = 0.1
    ) -> None:
        """平滑流表面（1D 平滑）。

        Args:
            surface: 流表面
            nsmooth: 平滑迭代次數
            smooth_factor: 平滑因子 (0-1)
        """
        for _ in range(nsmooth):
            # 平滑軸向坐標
            x_new = surface.x.copy()
            for i in range(1, surface.npoints - 1):
                avg = 0.5 * (surface.x[i - 1] + surface.x[i + 1])
                x_new[i] = (1.0 - smooth_factor) * surface.x[i] + smooth_factor * avg
            surface.x = x_new

            # 平滑徑向坐標
            r_new = surface.r.copy()
            for i in range(1, surface.npoints - 1):
                avg = 0.5 * (surface.r[i - 1] + surface.r[i + 1])
                r_new[i] = (1.0 - smooth_factor) * surface.r[i] + smooth_factor * avg
            surface.r = r_new

        # 重新計算衍生量
        surface.pitch_angle = self._calculate_pitch_angle(surface.x, surface.r)
        surface.s_dist = self._calculate_distance(surface.x, surface.r)

    def smooth_surface_2d(
        self,
        surface: StreamSurface,
        nsmooth: int = 5,
        smooth_factor: float = 0.1,
    ) -> None:
        """平滑流表面（2D 平滑，保持連續性）。

        Args:
            surface: 流表面
            nsmooth: 平滑迭代次數
            smooth_factor: 平滑因子 (0-1)
        """
        for _ in range(nsmooth):
            for i in range(1, surface.npoints - 1):
                # 計算相鄰點連線
                dx_prev = surface.x[i] - surface.x[i - 1]
                dr_prev = surface.r[i] - surface.r[i - 1]

                dx_next = surface.x[i + 1] - surface.x[i]
                dr_next = surface.r[i + 1] - surface.r[i]

                # 計算垂直方向
                # 平均切向量
                tx = dx_prev + dx_next
                tr = dr_prev + dr_next
                t_len = math.sqrt(tx * tx + tr * tr)

                if t_len > 1e-10:
                    # 單位切向量
                    tx /= t_len
                    tr /= t_len

                    # 垂直向量（順時針旋轉90度）
                    nx = tr
                    nr = -tx

                    # 沿垂直方向移動點
                    # 這保持流表面平滑而不產生折疊
                    offset = smooth_factor * 0.01  # 小的調整量
                    surface.x[i] += offset * nx
                    surface.r[i] += offset * nr

        # 重新計算衍生量
        surface.pitch_angle = self._calculate_pitch_angle(surface.x, surface.r)
        surface.s_dist = self._calculate_distance(surface.x, surface.r)

    def interpolate_surface(
        self, surface: StreamSurface, new_npoints: int
    ) -> StreamSurface:
        """插值流表面到新的點數。

        Args:
            surface: 原始流表面
            new_npoints: 新的點數

        Returns:
            插值後的流表面
        """
        # 使用流表面距離作為參數
        s_old = np.array(surface.s_dist)
        s_new = np.linspace(s_old[0], s_old[-1], new_npoints)

        # 插值各個量
        x_new = np.interp(s_new, s_old, surface.x).tolist()
        r_new = np.interp(s_new, s_old, surface.r).tolist()
        vm_ratio_new = np.interp(s_new, s_old, surface.vm_ratio).tolist()
        fblock_new = np.interp(s_new, s_old, surface.fblock).tolist()

        # 計算衍生量
        pitch_angle_new = self._calculate_pitch_angle(x_new, r_new)
        s_dist_new = self._calculate_distance(x_new, r_new)

        return StreamSurface(
            npoints=new_npoints,
            x=x_new,
            r=r_new,
            vm_ratio=vm_ratio_new,
            pitch_angle=pitch_angle_new,
            s_dist=s_dist_new,
            fblock=fblock_new,
        )

    def merge_surfaces(
        self, hub_surface: StreamSurface, tip_surface: StreamSurface
    ) -> tuple[StreamSurface, StreamSurface]:
        """合併多級的 hub 和 tip 流表面。

        Args:
            hub_surface: Hub 流表面
            tip_surface: Tip 流表面

        Returns:
            合併後的 (hub, tip) 流表面
        """
        # TODO: 實現多級流表面合併
        # - 移除重疊點
        # - 平滑連接處
        # - 確保連續性
        return hub_surface, tip_surface

    def create_mean_surface(
        self, hub_surface: StreamSurface, tip_surface: StreamSurface
    ) -> StreamSurface:
        """從 hub 和 tip 創建平均流表面。

        Args:
            hub_surface: Hub 流表面
            tip_surface: Tip 流表面

        Returns:
            平均流表面
        """
        # 確保點數相同
        if hub_surface.npoints != tip_surface.npoints:
            raise ValueError("Hub 和 Tip 流表面點數必須相同")

        # 計算平均值
        x_mean = [(h + t) / 2.0 for h, t in zip(hub_surface.x, tip_surface.x)]
        r_mean = [(h + t) / 2.0 for h, t in zip(hub_surface.r, tip_surface.r)]
        vm_ratio_mean = [
            (h + t) / 2.0 for h, t in zip(hub_surface.vm_ratio, tip_surface.vm_ratio)
        ]
        fblock_mean = [
            (h + t) / 2.0 for h, t in zip(hub_surface.fblock, tip_surface.fblock)
        ]

        # 計算衍生量
        pitch_angle_mean = self._calculate_pitch_angle(x_mean, r_mean)
        s_dist_mean = self._calculate_distance(x_mean, r_mean)

        return StreamSurface(
            npoints=hub_surface.npoints,
            x=x_mean,
            r=r_mean,
            vm_ratio=vm_ratio_mean,
            pitch_angle=pitch_angle_mean,
            s_dist=s_dist_mean,
            fblock=fblock_mean,
        )
