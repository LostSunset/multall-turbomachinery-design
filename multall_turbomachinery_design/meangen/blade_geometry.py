# -*- coding: utf-8 -*-
"""葉片幾何生成模組。

提供葉片厚度分布、角度分布和 Zweifel 係數計算功能。
"""

from __future__ import annotations

import math

from .data_structures import CONSTANTS, BladeRow, MachineType


class BladeGeometryGenerator:
    """葉片幾何生成器。"""

    def __init__(self, machine_type: MachineType) -> None:
        """初始化葉片幾何生成器。

        Args:
            machine_type: 機械類型（渦輪或壓縮機）
        """
        self.machine_type = machine_type
        self.deg2rad = CONSTANTS["DEG2RAD"]
        self.rad2deg = CONSTANTS["RAD2DEG"]

        # 設置默認厚度參數
        if machine_type == MachineType.TURBINE:
            self.default_tk_le = 0.04  # 前緣厚度比
            self.default_tk_te = 0.04  # 後緣厚度比
            self.default_tk_max_stator = 0.30  # 定子最大厚度比
            self.default_tk_max_rotor = 0.25  # 轉子最大厚度比
            self.default_zweifel = 0.85  # Zweifel 係數
            self.default_angle_exponent = 1.0  # 角度分布指數
        else:  # COMPRESSOR
            self.default_tk_le = 0.02
            self.default_tk_te = 0.01
            self.default_tk_max_stator = 0.10
            self.default_tk_max_rotor = 0.075
            self.default_zweifel = 0.50
            self.default_angle_exponent = 1.35

    def calculate_blade_thickness_distribution(
        self,
        axial_chord: float,
        tk_max: float,
        xtk_max: float = 0.45,
        tk_le: float | None = None,
        tk_te: float | None = None,
        npoints: int = 200,
    ) -> tuple[list[float], list[float]]:
        """計算葉片厚度分布。

        Args:
            axial_chord: 軸向弦長 [m]
            tk_max: 最大厚度比（相對於軸向弦長）
            xtk_max: 最大厚度位置（相對於軸向弦長，0-1）
            tk_le: 前緣厚度比（None 使用默認值）
            tk_te: 後緣厚度比（None 使用默認值）
            npoints: 分布點數

        Returns:
            (x_positions, thickness_values) - 沿弦的位置和厚度
        """
        if tk_le is None:
            tk_le = self.default_tk_le
        if tk_te is None:
            tk_te = self.default_tk_te

        # 無量綱位置 (0-1)
        xi = [i / (npoints - 1) for i in range(npoints)]

        thickness = []
        for x in xi:
            if x <= xtk_max:
                # 前半段：從前緣到最大厚度
                # 使用平滑多項式
                t_ratio = x / xtk_max if xtk_max > 0 else 0.0
                t = tk_le + (tk_max - tk_le) * (3 * t_ratio**2 - 2 * t_ratio**3)  # 平滑 S 曲線
            else:
                # 後半段：從最大厚度到後緣
                t_ratio = (x - xtk_max) / (1.0 - xtk_max) if xtk_max < 1.0 else 0.0
                t = tk_max + (tk_te - tk_max) * (3 * t_ratio**2 - 2 * t_ratio**3)

            thickness.append(t * axial_chord)

        # 實際位置
        x_positions = [x * axial_chord for x in xi]

        return x_positions, thickness

    def calculate_blade_angle_distribution(
        self,
        alpha_in: float,
        alpha_out: float,
        npoints: int = 200,
        exponent: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """計算葉片角度沿弦的分布。

        使用 tan(α) 線性變化的假設。

        Args:
            alpha_in: 進口角度 [度]
            alpha_out: 出口角度 [度]
            npoints: 分布點數
            exponent: 角度分布指數（None 使用默認值）

        Returns:
            (xi, angles) - 無量綱位置 (0-1) 和角度 [度]
        """
        if exponent is None:
            exponent = self.default_angle_exponent

        # 轉換為 tan
        tan_in = math.tan(alpha_in * self.deg2rad)
        tan_out = math.tan(alpha_out * self.deg2rad)

        xi = []
        angles = []

        for i in range(npoints):
            x = i / (npoints - 1)
            xi.append(x)

            # tan(α) = tan(α_in) + (tan(α_out) - tan(α_in)) * x^E
            tan_alpha = tan_in + (tan_out - tan_in) * (x**exponent)

            # 轉換回角度
            alpha = math.atan(tan_alpha) * self.rad2deg
            angles.append(alpha)

        return xi, angles

    def calculate_zweifel_coefficient(
        self,
        alpha_in: float,
        alpha_out: float,
        vm: float,
        u: float,
    ) -> float:
        """計算 Zweifel 係數。

        Args:
            alpha_in: 進口絕對角 [度]
            alpha_out: 出口絕對角 [度]
            vm: 子午速度 [m/s]
            u: 圓周速度 [m/s]

        Returns:
            Zweifel 係數
        """
        # 轉換為弧度
        alpha_in_rad = alpha_in * self.deg2rad
        alpha_out_rad = alpha_out * self.deg2rad

        # Zweifel 係數公式（簡化版）
        # Zw = 2 * |tan(α_in) - tan(α_out)| * cos²(α_out)
        tan_in = math.tan(alpha_in_rad)
        tan_out = math.tan(alpha_out_rad)
        cos_out = math.cos(alpha_out_rad)

        # 對於渦輪/壓縮機，角度可能反向
        delta_tan = abs(tan_in - tan_out)

        zw = 2.0 * delta_tan * cos_out * cos_out

        # 如果結果太小，使用默認值
        if zw < 0.1:
            zw = self.default_zweifel

        return zw

    def calculate_blade_number(
        self,
        radius: float,
        axial_chord: float,
        pitch_angle: float,
        zweifel: float | None = None,
    ) -> int:
        """計算葉片數。

        Args:
            radius: 半徑 [m]
            axial_chord: 軸向弦長 [m]
            pitch_angle: 俯仰角 [度]
            zweifel: Zweifel 係數（None 使用默認值）

        Returns:
            葉片數
        """
        if zweifel is None:
            zweifel = self.default_zweifel

        # 計算周長
        circumference = 2.0 * CONSTANTS["PI"] * radius

        # 俯仰角修正
        pitch_rad = pitch_angle * self.deg2rad
        cos_pitch = math.cos(pitch_rad) if abs(pitch_angle) < 85 else 0.1

        # 葉片數 = 周長 * cos(pitch) / (Zweifel * 弦長)
        n_blades = circumference * cos_pitch / (zweifel * axial_chord)

        # 四捨五入到最接近的整數
        n_blades = round(n_blades)

        # 確保葉片數在合理範圍內
        if n_blades < 5:
            n_blades = 5
        elif n_blades > 200:
            n_blades = 200

        return n_blades

    def apply_incidence_deviation(
        self,
        blade_angles: list[float],
        incidence: float,
        deviation: float,
    ) -> list[float]:
        """應用入射角和偏角到葉片角度。

        Args:
            blade_angles: 流角 [度]
            incidence: 入射角 [度]
            deviation: 偏角 [度]

        Returns:
            金屬角 [度]
        """
        # 前緣：金屬角 = 流角 + 入射角
        # 後緣：金屬角 = 流角 + 偏角
        metal_angles = blade_angles.copy()

        npoints = len(blade_angles)
        if npoints > 0:
            # 前緣應用入射角
            metal_angles[0] += incidence

            # 後緣應用偏角
            metal_angles[-1] += deviation

            # 中間點線性插值
            if npoints > 2:
                for i in range(1, npoints - 1):
                    frac = i / (npoints - 1)
                    angle_correction = incidence + (deviation - incidence) * frac
                    metal_angles[i] += angle_correction

        return metal_angles

    def create_blade_row(
        self,
        row_number: int,
        row_type: str,
        radius: float,
        axial_chord: float,
        alpha_in: float,
        alpha_out: float,
        beta_in: float,
        beta_out: float,
        rpm: float,
        pitch_angle: float = 0.0,
        incidence: float = 0.0,
        deviation: float = 0.0,
        zweifel: float | None = None,
    ) -> BladeRow:
        """創建葉片排。

        Args:
            row_number: 排號
            row_type: 排類型 ('R'=轉子, 'S'=定子)
            radius: 半徑 [m]
            axial_chord: 軸向弦長 [m]
            alpha_in: 進口絕對角 [度]
            alpha_out: 出口絕對角 [度]
            beta_in: 進口相對角 [度]
            beta_out: 出口相對角 [度]
            rpm: 轉速 [RPM]
            pitch_angle: 俯仰角 [度]
            incidence: 入射角 [度]
            deviation: 偏角 [度]
            zweifel: Zweifel 係數

        Returns:
            葉片排
        """
        # 計算葉片數
        n_blades = self.calculate_blade_number(radius, axial_chord, pitch_angle, zweifel)

        # 設置默認厚度參數
        if row_type == "R":  # 轉子
            tk_max = self.default_tk_max_rotor
            xtk_max = 0.40
        else:  # 定子
            tk_max = self.default_tk_max_stator
            xtk_max = 0.45

        return BladeRow(
            row_number=row_number,
            row_type=row_type,
            n_blades=n_blades,
            rpm=rpm,
            axial_chord=axial_chord,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            beta_in=beta_in,
            beta_out=beta_out,
            tk_max=tk_max,
            tk_le=self.default_tk_le,
            tk_te=self.default_tk_te,
            xtk_max=xtk_max,
            incidence=incidence,
            deviation=deviation,
        )

    def generate_blade_sections(
        self,
        blade_row: BladeRow,
        radii: list[float],
        frac_twist: float = 1.0,
    ) -> list[tuple[float, list[float], list[float]]]:
        """生成多個葉片截面。

        Args:
            blade_row: 葉片排
            radii: 截面半徑列表 [m]
            frac_twist: 扭轉比例 (0=無扭轉, 1=完全扭轉)

        Returns:
            截面列表：每個截面為 (半徑, x坐標, y坐標)
        """
        sections = []

        for r in radii:
            # TODO: 應用自由渦扭轉調整角度
            # 目前使用設計點角度

            # 生成角度分布
            xi, angles = self.calculate_blade_angle_distribution(
                blade_row.alpha_in, blade_row.alpha_out
            )

            # 應用入射角和偏角
            _metal_angles = self.apply_incidence_deviation(
                angles, blade_row.incidence, blade_row.deviation
            )

            # 生成厚度分布
            x_pos, thickness = self.calculate_blade_thickness_distribution(
                blade_row.axial_chord,
                blade_row.tk_max,
                blade_row.xtk_max,
                blade_row.tk_le,
                blade_row.tk_te,
            )

            # TODO: 從角度和厚度生成實際的葉片輪廓 (x, y) 坐標
            # 這需要積分角度來得到中心線，然後加上厚度

            sections.append((r, x_pos, thickness))

        return sections
