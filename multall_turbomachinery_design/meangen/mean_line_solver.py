# -*- coding: utf-8 -*-
"""平均線求解器。

執行 MEANGEN 平均線設計的主求解器。
"""

from __future__ import annotations

import math
from pathlib import Path

from .blade_geometry import BladeGeometryGenerator
from .data_structures import (
    CONSTANTS,
    FlowType,
    InputType,
    MeangenConfig,
    StageDesign,
)
from .gas_properties import PerfectGasCalculator
from .io_handler import MeangenInputReader, MeangenOutputWriter, StagenOutputWriter
from .stream_surface import StreamSurfaceGenerator
from .velocity_triangles import VelocityTriangleCalculator


class MeanLineSolver:
    """MEANGEN 平均線求解器。"""

    def __init__(self, config: MeangenConfig) -> None:
        """初始化求解器。

        Args:
            config: MEANGEN 配置
        """
        self.config = config

        # 初始化計算器
        self.gas_calc = PerfectGasCalculator(config.gas)
        self.velocity_calc = VelocityTriangleCalculator(config.machine_type)
        self.surface_gen = StreamSurfaceGenerator(config.flow_type)
        self.blade_gen = BladeGeometryGenerator(config.machine_type)

        # 常數
        self.deg2rad = CONSTANTS["DEG2RAD"]
        self.rad2deg = CONSTANTS["RAD2DEG"]

    def solve_stage(self, stage: StageDesign) -> None:
        """求解單級設計。

        Args:
            stage: 級設計
        """
        # 計算設計點參數
        r = stage.r_design
        omega = 2.0 * CONSTANTS["PI"] * self.config.rpm / 60.0  # rad/s
        u = omega * r  # 圓周速度 [m/s]

        # 根據輸入類型計算速度三角形
        if stage.input_type == InputType.TYPE_A:
            # Type A: 使用 phi, psi, reaction
            alpha_in, alpha_out, beta_in, beta_out = self.velocity_calc.calculate_type_a(
                stage.phi, stage.psi, stage.reaction, u
            )
        elif stage.input_type == InputType.TYPE_B:
            # Type B: 使用 phi 和指定角度
            alpha_in = stage.alpha_in
            beta_in = stage.beta_in
            _alpha_out = stage.alpha_out  # 級出口角度（用於定子設計）
            beta_out = stage.beta_out
        else:
            # 其他類型，使用默認值
            alpha_in = beta_in = beta_out = 0.0

        # 計算子午速度
        vm = stage.phi * u if stage.phi > 0 else 100.0

        # 創建轉子進出口速度三角形
        # 轉子進口：使用絕對角 alpha_in
        inlet = self.velocity_calc.create_velocity_triangle(u, vm, alpha_in, is_rotor=True)

        # 轉子出口：從相對角 beta_out 計算絕對角
        # tan(α) = tan(β) + u/vm
        alpha_rotor_out = math.atan(
            math.tan(beta_out * self.deg2rad) + u / vm
        ) * self.rad2deg

        outlet = self.velocity_calc.create_velocity_triangle(
            u, vm, alpha_rotor_out, is_rotor=True
        )

        # 計算流場
        # TODO: 實現完整的熱力學計算
        # - 計算出口總溫（考慮做功）
        # - 計算出口總壓（考慮效率）
        # - 計算馬赫數
        # - 檢查壅塞

        # 計算級性能
        stage.work_output = u * (inlet.vtheta - outlet.vtheta)  # 比功 [J/kg]
        stage.loading_coefficient = stage.work_output / (u * u)

        # 保存速度三角形
        stage.inlet_triangle = inlet
        stage.outlet_triangle = outlet

        # 生成流表面（設計點）
        if self.config.flow_type == FlowType.AXIAL:
            surface = self.surface_gen.generate_axial_surface(
                r_design=r,
                axial_chord_1=stage.axial_chord_1 if stage.axial_chord_1 else 0.05,
                axial_chord_2=stage.axial_chord_2 if stage.axial_chord_2 else 0.04,
                row_gap=stage.row_gap if stage.row_gap else 0.025,
                stage_gap=stage.stage_gap if stage.stage_gap else 0.05,
            )

            # 應用堵塞因子
            nle = 2  # 轉子前緣點索引
            nte = 3  # 轉子後緣點索引
            self.surface_gen.apply_blockage_factor(
                surface, stage.fblock_le, stage.fblock_te, nle, nte
            )

            stage.stream_surface = surface

        # 創建葉片排
        # 轉子
        stage.rotor = self.blade_gen.create_blade_row(
            row_number=stage.stage_number * 2 - 1,  # 奇數為轉子
            row_type="R",
            radius=r,
            axial_chord=stage.axial_chord_1 if stage.axial_chord_1 else 0.05,
            alpha_in=inlet.alpha,
            alpha_out=outlet.alpha,
            beta_in=inlet.beta,
            beta_out=outlet.beta,
            rpm=self.config.rpm,
            incidence=stage.ainc1 if stage.ainc1 else 0.0,
            deviation=stage.devn1 if stage.devn1 else 0.0,
        )

        # 定子
        stage.stator = self.blade_gen.create_blade_row(
            row_number=stage.stage_number * 2,  # 偶數為定子
            row_type="S",
            radius=r,
            axial_chord=stage.axial_chord_2 if stage.axial_chord_2 else 0.04,
            alpha_in=outlet.alpha,
            alpha_out=alpha_in,  # 定子出口恢復到進口條件
            beta_in=outlet.beta,
            beta_out=beta_in,
            rpm=0.0,  # 定子不旋轉
            incidence=stage.ainc2 if stage.ainc2 else 0.0,
            deviation=stage.devn2 if stage.devn2 else 0.0,
        )

    def solve_all_stages(self) -> None:
        """求解所有級。"""
        for stage in self.config.stages:
            self.solve_stage(stage)

    def calculate_overall_performance(self) -> dict[str, float]:
        """計算整體性能。

        Returns:
            性能參數字典
        """
        total_work = sum(stage.work_output for stage in self.config.stages)
        power = total_work * self.config.mass_flow / 1000.0  # kW

        # 計算壓比和溫比
        # TODO: 實現完整的多級疊加計算

        return {
            "total_work": total_work,  # J/kg
            "power": power,  # kW
            "mass_flow": self.config.mass_flow,  # kg/s
            "stages": self.config.nstages,
        }

    @classmethod
    def from_input_file(cls, input_file: str | Path) -> MeanLineSolver:
        """從輸入檔案創建求解器。

        Args:
            input_file: meangen.in 檔案路徑

        Returns:
            求解器實例
        """
        reader = MeangenInputReader()
        config = reader.read_config(input_file)
        return cls(config)

    def write_outputs(
        self, output_dir: str | Path, write_stagen: bool = True
    ) -> None:
        """寫入輸出檔案。

        Args:
            output_dir: 輸出目錄
            write_stagen: 是否寫入 stagen.dat
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 寫入 meangen.out
        meangen_out = output_dir / "meangen.out"
        writer = MeangenOutputWriter()
        writer.write_output_file(self.config, meangen_out)

        # 寫入 stagen.dat
        if write_stagen:
            stagen_dat = output_dir / "stagen.dat"
            stagen_writer = StagenOutputWriter()
            stagen_writer.write_stagen_file(self.config, stagen_dat)

    def run(self, output_dir: str | Path | None = None) -> dict[str, float]:
        """執行完整的平均線設計。

        Args:
            output_dir: 輸出目錄（None 則不寫入檔案）

        Returns:
            性能參數字典
        """
        # 求解所有級
        self.solve_all_stages()

        # 計算整體性能
        performance = self.calculate_overall_performance()

        # 寫入輸出
        if output_dir:
            self.write_outputs(output_dir)

        return performance
