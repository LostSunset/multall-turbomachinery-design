# -*- coding: utf-8 -*-
"""網格生成模組。

生成周向和跨向網格，並應用擴張比。
"""

from __future__ import annotations

import numpy as np

from .data_structures import BladeRow, GridParameters


class GridGenerator:
    """網格生成器。"""

    def __init__(self) -> None:
        """初始化網格生成器。"""
        pass

    def calculate_pitchwise_expansion(
        self, im: int, fp_rat: float, fp_max: float
    ) -> list[float]:
        """計算周向網格擴張比。

        使用對稱擴張策略，從前緣和後緣兩端向中心擴張。

        Args:
            im: 周向網格點數
            fp_rat: 周向擴張比
            fp_max: 周向最大擴張比

        Returns:
            周向擴張因子列表
        """
        imm1 = im - 1

        # 從前端計算擴張
        f1 = [1.0]
        for i in range(1, imm1):
            f1.append(f1[-1] * fp_rat)

        # 從後端計算擴張（反向）
        f2 = [f1[im - 2 - i] for i in range(imm1)]

        # 取兩者中的較小值，並限制最大值
        fp = []
        for i in range(imm1):
            f = min(f1[i], f2[i])
            f = min(f, fp_max)
            fp.append(f)

        return fp

    def calculate_spanwise_expansion(
        self, km: int, fr_rat: float, fr_max: float
    ) -> tuple[list[float], list[float]]:
        """計算跨向網格擴張比。

        使用對稱擴張策略，從 HUB 和 CASING 兩端向中心擴張。

        Args:
            km: 跨向網格點數
            fr_rat: 跨向擴張比
            fr_max: 跨向最大擴張比

        Returns:
            (跨向擴張因子列表, 累積擴張因子列表)
        """
        kmm1 = km - 1

        # 從 HUB 計算擴張
        f1 = [1.0]
        for k in range(1, kmm1):
            f1.append(f1[-1] * fr_rat)

        # 從 CASING 計算擴張（反向）
        f2 = [f1[km - 2 - k] for k in range(kmm1)]

        # 取兩者中的較小值，並限制最大值
        fr = []
        for k in range(kmm1):
            f = min(f1[k], f2[k])
            f = min(f, fr_max)
            fr.append(f)

        # 計算累積因子
        sumfr = [0.0]
        for k in range(1, km):
            sumfr.append(sumfr[-1] + fr[k - 1])

        return fr, sumfr

    def generate_pitchwise_grid(
        self, n_blades: int, pitch: float, grid_params: GridParameters
    ) -> np.ndarray:
        """生成周向網格坐標。

        Args:
            n_blades: 葉片數
            pitch: 葉片間距 [m]
            grid_params: 網格參數

        Returns:
            周向網格坐標 (im,) [m]
        """
        im = grid_params.im
        fp = self.calculate_pitchwise_expansion(im, grid_params.fp_rat, grid_params.fp_max)

        # 計算總擴張和
        sum_fp = sum(fp)

        # 生成網格坐標
        y_grid = np.zeros(im)
        y_grid[0] = 0.0

        for i in range(1, im):
            # 累積網格間距
            y_grid[i] = y_grid[i - 1] + (pitch / sum_fp) * fp[i - 1]

        # 中心化（質心在 0）
        y_center = y_grid[im // 2]
        y_grid -= y_center

        return y_grid

    def generate_spanwise_grid(
        self, r_hub: float, r_tip: float, grid_params: GridParameters
    ) -> np.ndarray:
        """生成跨向網格坐標。

        Args:
            r_hub: HUB 半徑 [m]
            r_tip: TIP 半徑 [m]
            grid_params: 網格參數

        Returns:
            跨向網格坐標（半徑）(km,) [m]
        """
        km = grid_params.km
        fr, sumfr = self.calculate_spanwise_expansion(
            km, grid_params.fr_rat, grid_params.fr_max
        )

        # 計算總跨度
        span = r_tip - r_hub

        # 生成網格坐標
        r_grid = np.zeros(km)
        r_grid[0] = r_hub

        # 歸一化累積因子
        sum_total = sumfr[-1] if sumfr[-1] > 0 else 1.0

        for k in range(1, km):
            # 從 HUB 到 TIP
            r_grid[k] = r_hub + span * (sumfr[k] / sum_total)

        return r_grid

    def generate_axial_grid(
        self,
        x_le: float,
        x_te: float,
        axial_chord: float,
        grid_params: GridParameters,
    ) -> tuple[np.ndarray, int, int]:
        """生成軸向網格坐標。

        在前緣上游、葉片上、後緣下游三個區域生成網格。

        Args:
            x_le: 前緣軸向坐標 [m]
            x_te: 後緣軸向坐標 [m]
            axial_chord: 軸向弦長 [m]
            grid_params: 網格參數

        Returns:
            (軸向網格坐標 [m], 前緣索引, 後緣索引)
        """
        nint_up = grid_params.nint_up
        nint_on = grid_params.nint_on
        nint_dn = grid_params.nint_dn

        # 上游區域（從 x_le - axial_chord 到 x_le）
        x_up = np.linspace(x_le - axial_chord, x_le, nint_up + 1)[:-1]

        # 葉片區域（從 x_le 到 x_te）
        x_on = np.linspace(x_le, x_te, nint_on + 1)

        # 下游區域（從 x_te 到 x_te + axial_chord）
        x_dn = np.linspace(x_te, x_te + axial_chord, nint_dn + 1)[1:]

        # 組合所有區域
        x_grid = np.concatenate([x_up, x_on, x_dn])

        # 前後緣索引
        j_le = nint_up
        j_te = nint_up + nint_on

        return x_grid, j_le, j_te

    def generate_blade_row_grid(
        self,
        blade_row: BladeRow,
        r_hub: float,
        r_tip: float,
        x_le: float,
        x_te: float,
        axial_chord: float,
    ) -> None:
        """為葉片排生成完整網格。

        將網格坐標存儲在 blade_row 中。

        Args:
            blade_row: 葉片排對象
            r_hub: HUB 半徑 [m]
            r_tip: TIP 半徑 [m]
            x_le: 前緣軸向坐標 [m]
            x_te: 後緣軸向坐標 [m]
            axial_chord: 軸向弦長 [m]
        """
        if blade_row.grid_params is None:
            blade_row.grid_params = GridParameters()

        grid_params = blade_row.grid_params

        # 周向間距
        pitch = 2.0 * np.pi * 0.5 * (r_hub + r_tip) / blade_row.n_blade

        # 生成網格
        _y_grid = self.generate_pitchwise_grid(blade_row.n_blade, pitch, grid_params)
        _r_grid = self.generate_spanwise_grid(r_hub, r_tip, grid_params)
        _x_grid, j_le, j_te = self.generate_axial_grid(
            x_le, x_te, axial_chord, grid_params
        )

        # 存儲索引
        blade_row.j_le = j_le
        blade_row.j_te = j_te
        blade_row.j_m = len(_x_grid)

    def calculate_spanwise_fractions(
        self, km: int, fr_rat: float, fr_max: float
    ) -> list[float]:
        """計算各跨向截面的分數位置（0=HUB, 1=TIP）。

        Args:
            km: 跨向網格點數
            fr_rat: 跨向擴張比
            fr_max: 跨向最大擴張比

        Returns:
            跨向分數位置列表
        """
        _fr, sumfr = self.calculate_spanwise_expansion(km, fr_rat, fr_max)

        # 歸一化
        sum_total = sumfr[-1] if sumfr[-1] > 0 else 1.0

        fractions = [s / sum_total for s in sumfr]

        return fractions
