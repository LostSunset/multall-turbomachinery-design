# -*- coding: utf-8 -*-
"""流線投影模組。

將 2D 葉片截面投影到 3D 流線表面。
"""

from __future__ import annotations

import math

import numpy as np

from .data_structures import (
    BladeProfile2D,
    BladeSection3D,
    StackingParameters,
    StreamSurface3D,
)


class StreamSurfaceProjector:
    """流線表面投影器。"""

    def __init__(self) -> None:
        """初始化投影器。"""
        pass

    def create_stream_surface(
        self, x_coords: list[float], r_coords: list[float]
    ) -> StreamSurface3D:
        """創建流線表面。

        Args:
            x_coords: 軸向坐標列表 [m]
            r_coords: 半徑坐標列表 [m]

        Returns:
            流線表面
        """
        n = len(x_coords)
        surface = StreamSurface3D(npoints=n)
        surface.x = x_coords.copy()
        surface.r = r_coords.copy()

        # 計算子午線距離
        s_merid = [0.0]
        for i in range(1, n):
            dx = x_coords[i] - x_coords[i - 1]
            dr = r_coords[i] - r_coords[i - 1]
            ds = math.sqrt(dx * dx + dr * dr)
            s_merid.append(s_merid[-1] + ds)

        surface.s_meridional = s_merid

        return surface

    def locate_leading_trailing_edges(
        self, surface: StreamSurface3D, le_x: float, te_x: float
    ) -> None:
        """定位前後緣在流線表面上的位置。

        Args:
            surface: 流線表面
            le_x: 前緣軸向坐標 [m]
            te_x: 後緣軸向坐標 [m]
        """
        # 插值找到前後緣的半徑和子午線距離
        surface.le_x = le_x
        surface.te_x = te_x

        # 線性插值找到半徑
        surface.le_r = np.interp(le_x, surface.x, surface.r)
        surface.te_r = np.interp(te_x, surface.x, surface.r)

        # 線性插值找到子午線距離
        surface.le_s = np.interp(le_x, surface.x, surface.s_meridional)
        surface.te_s = np.interp(te_x, surface.x, surface.s_meridional)

        # 子午弦長
        surface.chord_meridional = surface.te_s - surface.le_s

    def project_profile_to_surface(
        self,
        profile: BladeProfile2D,
        surface: StreamSurface3D,
        section_number: int,
        spanwise_fraction: float,
    ) -> BladeSection3D:
        """將 2D 葉片截面投影到 3D 流線表面。

        Args:
            profile: 2D 葉片截面
            surface: 流線表面
            section_number: 截面號
            spanwise_fraction: 跨向分數位置

        Returns:
            3D 葉片截面
        """
        section = BladeSection3D(
            section_number=section_number, spanwise_fraction=spanwise_fraction
        )

        # 縮放因子：從單位弦（profile）到子午弦（surface）
        scale = surface.chord_meridional / profile.chord_length

        # 使用中弧線坐標進行投影
        npoints = len(profile.x_camber)

        # 沿子午線的位置
        s_grid = [
            surface.le_s + scale * x for x in profile.x_camber
        ]

        # 插值到流線表面獲得 x 和 r
        x_grid = np.interp(s_grid, surface.s_meridional, surface.x).tolist()
        r_grid = np.interp(s_grid, surface.s_meridional, surface.r).tolist()

        # 周向坐標（初始為 2D 的 y 坐標）
        y_grid = [y * scale for y in profile.y_camber]

        # 厚度
        tk_grid = [tk * scale for tk in profile.thickness]

        # 填充截面
        section.x_grid = x_grid
        section.y_grid = y_grid
        section.r_grid = r_grid
        section.s_grid = s_grid
        section.tk_grid = tk_grid

        # 定位前後緣索引
        section.j_le = 0
        section.j_te = npoints - 1

        return section

    def calculate_centroid(self, section: BladeSection3D) -> None:
        """計算截面質心。

        使用梯形法則積分計算截面面積和質心。

        Args:
            section: 3D 葉片截面
        """
        j_le = section.j_le
        j_te = section.j_te

        sum_area = 0.0
        sum_x = 0.0
        sum_y = 0.0

        for j in range(j_le + 1, j_te + 1):
            # 梯形面積
            area = 0.5 * (section.tk_grid[j] + section.tk_grid[j - 1]) * (
                section.x_grid[j] - section.x_grid[j - 1]
            )

            sum_area += area
            sum_x += area * 0.5 * (section.x_grid[j] + section.x_grid[j - 1])

            # Y 質心（考慮厚度偏移）
            y_mid = 0.5 * (section.y_grid[j] + section.y_grid[j - 1])
            tk_mid = 0.25 * (section.tk_grid[j] + section.tk_grid[j - 1])
            sum_y += area * (y_mid - tk_mid)

        if sum_area > 0:
            section.x_centroid = sum_x / sum_area
            section.y_centroid = sum_y / sum_area
        else:
            # 如果面積為 0，使用中點
            mid_idx = (j_le + j_te) // 2
            section.x_centroid = section.x_grid[mid_idx]
            section.y_centroid = section.y_grid[mid_idx]

    def convert_r_theta_to_cartesian(self, section: BladeSection3D) -> None:
        """將 R-THETA 坐標轉換為笛卡爾周向坐標。

        Args:
            section: 3D 葉片截面
        """
        # 計算累積周向角
        theta_new = [0.0]
        for j in range(1, len(section.x_grid)):
            # 使用平均半徑計算角度增量
            r_avg = 0.5 * (section.r_grid[j] + section.r_grid[j - 1])
            dy = section.y_grid[j] - section.y_grid[j - 1]

            if r_avg > 1e-10:
                d_theta = 2.0 * dy / (section.r_grid[j] + section.r_grid[j - 1])
            else:
                d_theta = 0.0

            theta_new.append(theta_new[-1] + d_theta)

        # 找到質心對應的角度
        # 插值找到質心 x 坐標對應的點
        j_cent = 0
        for j in range(len(section.x_grid) - 1):
            if section.x_grid[j] <= section.x_centroid <= section.x_grid[j + 1]:
                j_cent = j
                break

        theta_cent = theta_new[j_cent]

        # 相對於質心的周向坐標
        for j in range(len(section.x_grid)):
            section.y_grid[j] = section.r_grid[j] * (theta_new[j] - theta_cent)

    def apply_stacking(
        self,
        section: BladeSection3D,
        stacking: StackingParameters,
        surface: StreamSurface3D,
    ) -> None:
        """應用 3D 堆疊變換。

        Args:
            section: 3D 葉片截面
            stacking: 堆疊參數
            surface: 流線表面（用於重新插值半徑）
        """
        # 向 HUB 中心堆疊
        if stacking.f_centroid <= 1.01:
            dx_cent = stacking.f_centroid * (
                stacking.x_centroid_hub - section.x_centroid
            )
            dy_cent = stacking.f_centroid * (
                stacking.y_centroid_hub - section.y_centroid
            )

            for j in range(len(section.x_grid)):
                section.x_grid[j] += dx_cent
                section.y_grid[j] += dy_cent

            section.x_centroid += dx_cent
            section.y_centroid += dy_cent

        # 計算弦向量
        x_le = section.x_grid[section.j_le]
        y_le = section.y_grid[section.j_le]
        x_te = section.x_grid[section.j_te]
        y_te = section.y_grid[section.j_te]

        x_dif = x_te - x_le
        y_dif = y_te - y_le

        # 計算移動量
        delta_x = (
            -stacking.f_sweep * x_dif
            - stacking.f_lean * y_dif
            + stacking.f_axial * x_dif
        )
        delta_y = (
            -stacking.f_sweep * y_dif
            + stacking.f_lean * x_dif
            + stacking.f_tang * x_dif
        )

        # 固定點坐標
        x_const = x_le + stacking.f_const * x_dif

        # 應用縮放和移動
        for j in range(len(section.x_grid)):
            section.x_grid[j] = x_const + stacking.f_scale * (
                section.x_grid[j] - x_const + delta_x
            )
            section.y_grid[j] = stacking.f_scale * (section.y_grid[j] + delta_y)

        # 重新插值半徑（僅對軸對稱機械）
        # 檢查是否為軸對稱（半徑變化相對於軸向變化較小）
        r_dif = section.r_grid[section.j_te] - section.r_grid[section.j_le]
        if abs(r_dif) <= 0.5 * abs(x_dif):
            # 軸對稱，重新插值半徑
            section.r_grid = np.interp(
                section.x_grid, surface.x, surface.r
            ).tolist()
