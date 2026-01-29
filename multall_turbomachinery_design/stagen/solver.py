# -*- coding: utf-8 -*-
"""STAGEN 主求解器。

整合所有組件，執行完整的 3D 葉片幾何生成流程。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from .blade_profile import BladeProfileGenerator
from .data_structures import (
    BladeProfile2D,
    BladeRow,
    BladeSection3D,
    GridParameters,
    StackingParameters,
    StagenConfig,
    StreamSurface3D,
    ThicknessParameters,
)
from .grid_generator import GridGenerator
from .io_handler import StagenInputReader, StagenOutputWriter
from .projection import StreamSurfaceProjector

if TYPE_CHECKING:
    from collections.abc import Sequence


class StagenSolver:
    """STAGEN 主求解器。

    整合葉片截面生成、流線投影、堆疊變換和網格生成功能，
    執行完整的 3D 葉片幾何生成流程。
    """

    def __init__(self, config: StagenConfig) -> None:
        """初始化求解器。

        Args:
            config: STAGEN 配置
        """
        self.config = config
        self.profile_gen = BladeProfileGenerator()
        self.projector = StreamSurfaceProjector()
        self.grid_gen = GridGenerator()

        # 結果存儲
        self._profiles: dict[int, list[BladeProfile2D]] = {}
        self._surfaces: dict[int, list[StreamSurface3D]] = {}
        self._sections: dict[int, list[BladeSection3D]] = {}

    @classmethod
    def from_input_file(cls, input_file: str | Path) -> StagenSolver:
        """從輸入文件創建求解器。

        Args:
            input_file: stagen.dat 文件路徑

        Returns:
            求解器實例
        """
        reader = StagenInputReader(input_file)
        config = reader.read()
        return cls(config)

    def generate_blade_profile(
        self,
        camber_slope: Sequence[float],
        x_fractions: Sequence[float],
        thickness_params: ThicknessParameters,
        npoints: int = 200,
    ) -> BladeProfile2D:
        """生成單個 2D 葉片截面。

        Args:
            camber_slope: 中弧線斜率（dy/dx）
            x_fractions: 對應的軸向分數位置
            thickness_params: 厚度參數
            npoints: 網格點數

        Returns:
            2D 葉片截面
        """
        profile = self.profile_gen.generate_from_camber_thickness(
            camber_slope=list(camber_slope),
            x_fractions=list(x_fractions),
            thickness_params=thickness_params,
            npoints=npoints,
        )
        return profile

    def create_stream_surface(
        self,
        x_coords: Sequence[float],
        r_coords: Sequence[float],
        le_x: float,
        te_x: float,
    ) -> StreamSurface3D:
        """創建流線表面並定位前後緣。

        Args:
            x_coords: 軸向坐標列表 [m]
            r_coords: 半徑坐標列表 [m]
            le_x: 前緣軸向坐標 [m]
            te_x: 後緣軸向坐標 [m]

        Returns:
            流線表面
        """
        surface = self.projector.create_stream_surface(list(x_coords), list(r_coords))
        self.projector.locate_leading_trailing_edges(surface, le_x, te_x)
        return surface

    def project_to_3d(
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
        section = self.projector.project_profile_to_surface(
            profile=profile,
            surface=surface,
            section_number=section_number,
            spanwise_fraction=spanwise_fraction,
        )

        # 計算質心
        self.projector.calculate_centroid(section)

        # R-THETA 轉換
        self.projector.convert_r_theta_to_cartesian(section)

        return section

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
            surface: 流線表面
        """
        self.projector.apply_stacking(section, stacking, surface)

    def generate_blade_row(
        self,
        row_number: int,
        row_type: str,
        n_blades: int,
        rpm: float,
        profiles: list[BladeProfile2D],
        surfaces: list[StreamSurface3D],
        stacking_params: list[StackingParameters] | None = None,
    ) -> BladeRow:
        """生成完整的葉片排。

        Args:
            row_number: 排號
            row_type: 類型（'R' = 轉子, 'S' = 定子）
            n_blades: 葉片數
            rpm: 轉速 [RPM]
            profiles: 各截面的 2D 葉片輪廓
            surfaces: 各截面的流線表面
            stacking_params: 各截面的堆疊參數（可選）

        Returns:
            葉片排
        """
        blade_row = BladeRow(
            row_number=row_number,
            row_type=row_type,
            n_blade=n_blades,
            rpm=rpm,
            grid_params=GridParameters(
                im=self.config.grid_params.im,
                km=self.config.grid_params.km,
                fp_rat=self.config.grid_params.fp_rat,
                fp_max=self.config.grid_params.fp_max,
                fr_rat=self.config.grid_params.fr_rat,
                fr_max=self.config.grid_params.fr_max,
            ),
        )

        nosect = len(profiles)
        spanwise_fractions = self.grid_gen.calculate_spanwise_fractions(
            nosect,
            self.config.grid_params.fr_rat,
            self.config.grid_params.fr_max,
        )

        # 為每個截面生成 3D 幾何
        hub_section: BladeSection3D | None = None

        for i in range(nosect):
            profile = profiles[i]
            surface = surfaces[i]
            span_frac = spanwise_fractions[i]

            # 投影到 3D
            section = self.project_to_3d(
                profile=profile,
                surface=surface,
                section_number=i + 1,
                spanwise_fraction=span_frac,
            )

            # 保存 HUB 截面作為堆疊參考
            if i == 0:
                hub_section = section

            # 應用堆疊變換
            if stacking_params is not None and i < len(stacking_params):
                stacking = stacking_params[i]
                # 設置 HUB 質心作為參考
                if hub_section is not None:
                    stacking.x_centroid_hub = hub_section.x_centroid
                    stacking.y_centroid_hub = hub_section.y_centroid
                self.apply_stacking(section, stacking, surface)

            blade_row.sections.append(section)

        # 設置前後緣索引
        if blade_row.sections:
            blade_row.j_le = blade_row.sections[0].j_le
            blade_row.j_te = blade_row.sections[0].j_te
            blade_row.j_m = len(blade_row.sections[0].x_grid)

        return blade_row

    def solve_blade_row(
        self,
        row_number: int,
        row_type: str,
        n_blades: int,
        rpm: float,
        x_hub: list[float],
        r_hub: list[float],
        x_tip: list[float],
        r_tip: list[float],
        le_x: float,
        te_x: float,
        camber_slopes: list[list[float]],
        x_fractions: list[float],
        thickness_params: ThicknessParameters,
        stacking_params: list[StackingParameters] | None = None,
    ) -> BladeRow:
        """求解單個葉片排。

        這是一個高階方法，整合了所有步驟：
        1. 為每個跨向截面生成 2D 葉片輪廓
        2. 創建流線表面
        3. 投影到 3D
        4. 應用堆疊變換

        Args:
            row_number: 排號
            row_type: 類型（'R' = 轉子, 'S' = 定子）
            n_blades: 葉片數
            rpm: 轉速 [RPM]
            x_hub: HUB 流線軸向坐標
            r_hub: HUB 流線半徑坐標
            x_tip: TIP 流線軸向坐標
            r_tip: TIP 流線半徑坐標
            le_x: 前緣軸向坐標
            te_x: 後緣軸向坐標
            camber_slopes: 各截面的中弧線斜率
            x_fractions: 斜率對應的軸向分數位置
            thickness_params: 厚度參數
            stacking_params: 堆疊參數（可選）

        Returns:
            葉片排
        """
        nosect = self.config.nosect
        profiles: list[BladeProfile2D] = []
        surfaces: list[StreamSurface3D] = []

        # 計算跨向分數位置
        spanwise_fractions = self.grid_gen.calculate_spanwise_fractions(
            nosect,
            self.config.grid_params.fr_rat,
            self.config.grid_params.fr_max,
        )

        for i in range(nosect):
            span_frac = spanwise_fractions[i]

            # 插值得到該截面的流線坐標
            x_coords = self._interpolate_streamline(x_hub, x_tip, span_frac)
            r_coords = self._interpolate_streamline(r_hub, r_tip, span_frac)

            # 插值得到該截面的中弧線斜率（如果有多組）
            if len(camber_slopes) == nosect:
                slopes = camber_slopes[i]
            else:
                # 只有一組斜率，所有截面使用相同斜率
                slopes = camber_slopes[0]

            # 生成 2D 葉片輪廓
            profile = self.generate_blade_profile(
                camber_slope=slopes,
                x_fractions=x_fractions,
                thickness_params=thickness_params,
            )
            profiles.append(profile)

            # 創建流線表面
            surface = self.create_stream_surface(
                x_coords=x_coords,
                r_coords=r_coords,
                le_x=le_x,
                te_x=te_x,
            )
            surfaces.append(surface)

        # 生成完整葉片排
        blade_row = self.generate_blade_row(
            row_number=row_number,
            row_type=row_type,
            n_blades=n_blades,
            rpm=rpm,
            profiles=profiles,
            surfaces=surfaces,
            stacking_params=stacking_params,
        )

        # 存儲中間結果
        self._profiles[row_number] = profiles
        self._surfaces[row_number] = surfaces
        self._sections[row_number] = blade_row.sections

        return blade_row

    def _interpolate_streamline(
        self,
        coords_hub: list[float],
        coords_tip: list[float],
        span_frac: float,
    ) -> list[float]:
        """線性插值流線坐標。

        Args:
            coords_hub: HUB 坐標
            coords_tip: TIP 坐標
            span_frac: 跨向分數（0=HUB, 1=TIP）

        Returns:
            插值後的坐標
        """
        result = []
        for h, t in zip(coords_hub, coords_tip, strict=False):
            result.append(h + span_frac * (t - h))
        return result

    def solve(self) -> None:
        """執行完整求解流程。

        對配置中的所有葉片排執行 3D 幾何生成。
        """
        for blade_row in self.config.blade_rows:
            # 如果截面已存在，跳過
            if blade_row.sections:
                continue

            # 否則需要從輸入數據生成
            # 這裡假設 blade_row 已有基本參數，
            # 實際使用時需要調用 solve_blade_row() 提供完整參數

    def write_outputs(self, output_dir: str | Path) -> None:
        """寫入輸出文件。

        Args:
            output_dir: 輸出目錄
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        writer = StagenOutputWriter(output_dir)
        writer.write_stage_old(self.config)
        writer.write_stage_new(self.config)
        writer.write_stagen_output(self.config)

    def run(self, output_dir: str | Path | None = None) -> dict[str, object]:
        """執行完整的 3D 葉片幾何生成。

        Args:
            output_dir: 輸出目錄（None 則不寫入文件）

        Returns:
            結果摘要字典
        """
        # 執行求解
        self.solve()

        # 統計結果
        total_sections = sum(len(row.sections) for row in self.config.blade_rows)
        total_rows = len(self.config.blade_rows)

        result = {
            "nrows": total_rows,
            "total_sections": total_sections,
            "nosect": self.config.nosect,
            "grid_im": self.config.grid_params.im,
            "grid_km": self.config.grid_params.km,
        }

        # 寫入輸出
        if output_dir:
            self.write_outputs(output_dir)
            result["output_dir"] = str(output_dir)

        return result

    def get_blade_coordinates(self, row_number: int) -> dict[str, list[list[list[float]]]]:
        """獲取葉片排的 3D 坐標。

        Args:
            row_number: 排號

        Returns:
            坐標字典，包含 x, y, r, thickness
        """
        if row_number > len(self.config.blade_rows):
            msg = f"排號 {row_number} 不存在"
            raise ValueError(msg)

        blade_row = self.config.blade_rows[row_number - 1]

        coords: dict[str, list[list[list[float]]]] = {
            "x": [],
            "y": [],
            "r": [],
            "thickness": [],
        }

        for section in blade_row.sections:
            coords["x"].append([section.x_grid])
            coords["y"].append([section.y_grid])
            coords["r"].append([section.r_grid])
            coords["thickness"].append([section.tk_grid])

        return coords


def create_simple_blade_row(
    row_type: str = "R",
    n_blades: int = 30,
    rpm: float = 3000.0,
    r_hub: float = 0.25,
    r_tip: float = 0.35,
    le_x: float = 0.02,
    te_x: float = 0.08,
    inlet_angle: float = 30.0,
    outlet_angle: float = -30.0,
    tk_max: float = 0.08,
    nosect: int = 5,
) -> StagenSolver:
    """創建簡單葉片排的便捷函數。

    Args:
        row_type: 類型（'R' = 轉子, 'S' = 定子）
        n_blades: 葉片數
        rpm: 轉速 [RPM]
        r_hub: HUB 半徑 [m]
        r_tip: TIP 半徑 [m]
        le_x: 前緣軸向坐標 [m]
        te_x: 後緣軸向坐標 [m]
        inlet_angle: 進口角 [度]
        outlet_angle: 出口角 [度]
        tk_max: 最大厚度比
        nosect: 截面數

    Returns:
        配置好的求解器
    """
    # 創建配置
    config = StagenConfig(
        rgas=287.0,
        gamma=1.4,
        nrows=1,
        nosect=nosect,
        grid_params=GridParameters(
            im=37,
            km=nosect,
            fp_rat=1.25,
            fp_max=20.0,
            fr_rat=1.25,
            fr_max=20.0,
        ),
    )

    solver = StagenSolver(config)

    # 從角度計算中弧線斜率
    deg2rad = math.pi / 180.0
    inlet_slope = math.tan(inlet_angle * deg2rad)
    outlet_slope = math.tan(outlet_angle * deg2rad)

    camber_slopes = [[inlet_slope, 0.0, outlet_slope]]
    x_fractions = [0.0, 0.5, 1.0]

    # 創建厚度參數
    thickness_params = ThicknessParameters(
        tk_le=0.02,
        tk_te=0.01,
        tk_max=tk_max,
        xtk_max=0.40,
    )

    # 創建流線坐標（軸向流）
    n_points = 6
    x_coords = [i * (te_x - le_x + 0.04) / (n_points - 1) for i in range(n_points)]
    r_hub_coords = [r_hub] * n_points
    r_tip_coords = [r_tip] * n_points

    # 求解葉片排
    blade_row = solver.solve_blade_row(
        row_number=1,
        row_type=row_type,
        n_blades=n_blades,
        rpm=rpm if row_type == "R" else 0.0,
        x_hub=x_coords,
        r_hub=r_hub_coords,
        x_tip=x_coords,
        r_tip=r_tip_coords,
        le_x=le_x,
        te_x=te_x,
        camber_slopes=camber_slopes,
        x_fractions=x_fractions,
        thickness_params=thickness_params,
    )

    config.blade_rows.append(blade_row)

    return solver
