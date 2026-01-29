# -*- coding: utf-8 -*-
"""2D 葉片截面生成模組。

提供從參數化定義生成 2D 葉片截面的功能。
"""

from __future__ import annotations

import math

import numpy as np

from .data_structures import BladeProfile2D, STAGEN_CONSTANTS, ThicknessParameters


class BladeProfileGenerator:
    """2D 葉片截面生成器。"""

    def __init__(self) -> None:
        """初始化生成器。"""
        self.pi = STAGEN_CONSTANTS["PI"]
        self.deg2rad = STAGEN_CONSTANTS["DEG2RAD"]
        self.rad2deg = STAGEN_CONSTANTS["RAD2DEG"]

    def generate_from_camber_thickness(
        self,
        camber_slope: list[float],
        x_fractions: list[float],
        thickness_params: ThicknessParameters,
        npoints: int = 200,
    ) -> BladeProfile2D:
        """從中弧線斜率和厚度參數生成葉片截面（INTYPE=1）。

        Args:
            camber_slope: 中弧線斜率 dy/dx 列表
            x_fractions: 中弧線斜率對應的軸向分數位置 (0-1)
            thickness_params: 厚度分佈參數
            npoints: 輸出點數

        Returns:
            2D 葉片截面
        """
        profile = BladeProfile2D()

        # 在 npoints 個點上插值中弧線斜率
        x_uniform = np.linspace(0.0, 1.0, npoints)
        slope_interp = np.interp(x_uniform, x_fractions, camber_slope)

        # 從斜率積分得到中弧線坐標
        y_camber = np.zeros(npoints)
        x_camber = np.zeros(npoints)
        x_camber[0] = 0.0
        y_camber[0] = 0.0

        for i in range(1, npoints):
            dx = x_uniform[i] - x_uniform[i - 1]
            # 使用梯形法則積分
            dy = 0.5 * (slope_interp[i] + slope_interp[i - 1]) * dx
            x_camber[i] = x_uniform[i]
            y_camber[i] = y_camber[i - 1] + dy

        # 計算厚度分佈
        thickness = self._calculate_thickness_distribution(
            x_uniform, thickness_params
        )

        # 計算前後緣調整因子
        faclete = self._calculate_le_te_factor(x_uniform, thickness_params)

        # 應用調整因子到厚度
        thickness_adjusted = thickness * faclete

        # 添加厚度到中弧線以形成上下表面
        x_upper, y_upper, x_lower, y_lower = self._add_thickness_to_camber(
            x_camber, y_camber, slope_interp, thickness_adjusted, thickness_params
        )

        # 填充 profile
        profile.x_camber = x_camber.tolist()
        profile.y_camber = y_camber.tolist()
        profile.camber_slope = slope_interp.tolist()
        profile.thickness = thickness_adjusted.tolist()
        profile.x_upper = x_upper.tolist()
        profile.y_upper = y_upper.tolist()
        profile.x_lower = x_lower.tolist()
        profile.y_lower = y_lower.tolist()

        return profile

    def _calculate_thickness_distribution(
        self, x: np.ndarray, params: ThicknessParameters
    ) -> np.ndarray:
        """計算厚度分佈。

        使用功率變換將最大厚度位置映射到指定位置，
        並使用高斯型函數形成厚度分佈。

        Args:
            x: 軸向分數位置 (0-1)
            params: 厚度參數

        Returns:
            厚度分佈數組
        """
        # 功率變換以定位最大厚度
        power = math.log(0.5) / math.log(params.xtk_max)
        x_trans = np.power(x, power)

        # 線性基礎厚度（從前緣到後緣）
        tk_linear = params.tk_le + x * (params.tk_te - params.tk_le)

        # 額外厚度（在最大厚度位置達到峰值）
        tk_add_max = params.tk_max - (
            params.tk_le + params.xtk_max * (params.tk_te - params.tk_le)
        )

        # 使用高斯型函數添加額外厚度
        exponent = params.tk_type
        tk_add = tk_add_max * (
            1.0 - np.power(np.abs(x_trans - 0.5) / 0.5, exponent)
        )

        # 總厚度
        thickness = tk_linear + tk_add

        return thickness

    def _calculate_le_te_factor(
        self, x: np.ndarray, params: ThicknessParameters
    ) -> np.ndarray:
        """計算前後緣調整因子。

        在前後緣附近使用橢圓形狀減小厚度。

        Args:
            x: 軸向分數位置 (0-1)
            params: 厚度參數

        Returns:
            調整因子數組 (0-1)
        """
        faclete = np.ones_like(x)

        # 前緣調整
        if params.xmod_le > 0.0001:
            x_le = x / params.xmod_le
            mask_le = x_le <= 1.0
            x_le_adjusted = np.abs(x_le - 1.0)
            faclete[mask_le] *= np.sqrt(
                1.0 - np.power(x_le_adjusted[mask_le], params.le_exp)
            )

        # 後緣調整
        if params.xmod_te > 0.001:
            x_te = (1.0 - x) / params.xmod_te
            mask_te = x_te <= 1.0
            x_te_adjusted = x_te - 1.0
            faclete[mask_te] *= np.sqrt(1.0 - np.power(x_te_adjusted[mask_te], 2.0))

        return faclete

    def _add_thickness_to_camber(
        self,
        x_camber: np.ndarray,
        y_camber: np.ndarray,
        slope: np.ndarray,
        thickness: np.ndarray,
        params: ThicknessParameters,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """添加厚度到中弧線以形成上下表面。

        Args:
            x_camber: 中弧線 X 坐標
            y_camber: 中弧線 Y 坐標
            slope: 中弧線斜率 dy/dx
            thickness: 厚度分佈
            params: 厚度參數

        Returns:
            (x_upper, y_upper, x_lower, y_lower)
        """
        # 計算中弧線的法向
        slope_angle = np.arctan(slope)

        # 垂直於中弧線的厚度分量
        half_tk = 0.5 * thickness
        d_perp = half_tk * np.cos(slope_angle)

        # 根據 f_perp 參數，部分垂直、部分豎直
        f_perp = params.f_perp
        f_perp1 = 1.0 - f_perp

        # 上表面
        x_upper = x_camber - f_perp * d_perp * np.sin(slope_angle)
        y_upper = (
            y_camber
            + f_perp * d_perp * np.cos(slope_angle)
            + f_perp1 * half_tk
        )

        # 下表面
        x_lower = x_camber + f_perp * d_perp * np.sin(slope_angle)
        y_lower = (
            y_camber
            - f_perp * d_perp * np.cos(slope_angle)
            - f_perp1 * half_tk
        )

        return x_upper, y_upper, x_lower, y_lower

    def smooth_curve(
        self, y_values: list[float], n_smooth: int = 5, smooth_factor: float = 0.5
    ) -> list[float]:
        """平滑曲線數據。

        使用加權平均和曲率保持進行迭代平滑。

        Args:
            y_values: 要平滑的 Y 值列表
            n_smooth: 平滑迭代次數
            smooth_factor: 平滑因子 (0-1)

        Returns:
            平滑後的 Y 值列表
        """
        y = np.array(y_values, dtype=float)
        n = len(y)

        for _ in range(n_smooth):
            # 計算平均值
            avg = np.zeros_like(y)
            avg[1:-1] = 0.5 * (y[:-2] + y[2:])
            avg[0] = y[0]
            avg[-1] = y[-1]

            # 計算曲率
            curv = y - avg

            # 平滑曲率
            curv_smooth = np.zeros_like(curv)
            curv_smooth[1:-1] = 0.5 * (curv[:-2] + curv[2:])
            curv_smooth[0] = curv[0]
            curv_smooth[-1] = curv[-1]

            # 應用平滑
            y_smooth = avg + 0.5 * curv_smooth
            y = (1.0 - smooth_factor) * y + smooth_factor * y_smooth

        return y.tolist()
