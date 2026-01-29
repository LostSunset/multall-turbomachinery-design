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
    FlowState,
    FlowType,
    InputType,
    MachineType,
    MeangenConfig,
    StageDesign,
    StageThermodynamics,
    VelocityTriangle,
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
        alpha_rotor_out = math.atan(math.tan(beta_out * self.deg2rad) + u / vm) * self.rad2deg

        outlet = self.velocity_calc.create_velocity_triangle(u, vm, alpha_rotor_out, is_rotor=True)

        # 計算級性能
        stage.work_output = u * (inlet.vtheta - outlet.vtheta)  # 比功 [J/kg]
        stage.loading_coefficient = stage.work_output / (u * u)

        # 計算熱力學狀態
        stage.thermodynamics = self._calculate_stage_thermodynamics(stage, inlet, outlet)

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

    def _calculate_stage_thermodynamics(
        self,
        stage: StageDesign,
        inlet: VelocityTriangle,
        outlet: VelocityTriangle,
    ) -> StageThermodynamics:
        """計算級熱力學狀態。

        Args:
            stage: 級設計
            inlet: 進口速度三角形
            outlet: 出口速度三角形

        Returns:
            級熱力學狀態
        """
        # 進口總條件（來自配置）
        poin = self.config.gas.poin * 1e5  # bar -> Pa
        toin = self.config.gas.toin  # K

        # 進口總焓和熵
        hoin = self.gas_calc.calculate_enthalpy_from_temperature(toin)
        sin = 0.0  # 參考熵

        # 進口速度
        v_in = math.sqrt(inlet.vm**2 + inlet.vtheta**2)

        # 計算進口靜態條件
        inlet_state = self.gas_calc.calculate_properties(
            ho=hoin, s=sin, v=v_in, poin=poin, hoin=hoin, sin=sin
        )

        # 出口速度
        v_out = math.sqrt(outlet.vm**2 + outlet.vtheta**2)

        # 出口總焓（考慮做功）
        # 對於渦輪，做功為正，總焓減少
        # 對於壓縮機，做功為負（輸入功），總焓增加
        if self.config.machine_type == MachineType.TURBINE:
            ho_out = hoin - stage.work_output
        else:  # 壓縮機
            ho_out = hoin + abs(stage.work_output)

        # 計算等熵出口狀態（理想情況）
        # 等熵過程：熵不變
        outlet_isentropic = self.gas_calc.calculate_properties(
            ho=ho_out, s=sin, v=v_out, poin=poin, hoin=hoin, sin=sin
        )

        # 實際出口狀態（考慮效率）
        eta = stage.efficiency

        if self.config.machine_type == MachineType.TURBINE:
            # 渦輪：實際焓降 = 效率 × 等熵焓降
            # 但我們用的是實際做功，所以需要計算實際熵增
            dh_isentropic = hoin - outlet_isentropic.ho
            dh_actual = stage.work_output

            # 實際出口總焓
            ho_actual = hoin - dh_actual

            # 計算熵增（損失）
            if eta > 0 and eta < 1:
                # 損失功
                loss_work = dh_actual * (1 / eta - 1)
                # 熵增 ≈ 損失功 / 平均溫度
                t_avg = 0.5 * (inlet_state.to + outlet_isentropic.to)
                ds = loss_work / t_avg if t_avg > 0 else 0.0
            else:
                ds = 0.0

            s_out = sin + ds

        else:  # 壓縮機
            # 壓縮機：等熵功 = 效率 × 實際功
            dh_actual = abs(stage.work_output)
            dh_isentropic = eta * dh_actual

            ho_actual = hoin + dh_actual

            # 計算熵增
            if eta > 0:
                loss_work = dh_actual - dh_isentropic
                t_avg = 0.5 * (inlet_state.to + outlet_isentropic.to)
                ds = loss_work / t_avg if t_avg > 0 else 0.0
            else:
                ds = 0.0

            s_out = sin + ds

        # 計算實際出口狀態
        outlet_state = self.gas_calc.calculate_properties(
            ho=ho_actual, s=s_out, v=v_out, poin=poin, hoin=hoin, sin=sin
        )

        # 計算壓比和溫比
        if inlet_state.po > 0:
            pressure_ratio = outlet_state.po / inlet_state.po
        else:
            pressure_ratio = 1.0

        if inlet_state.to > 0:
            temperature_ratio = outlet_state.to / inlet_state.to
        else:
            temperature_ratio = 1.0

        # 壅塞檢查
        is_choked = outlet_state.mach >= 1.0 or inlet_state.mach >= 1.0

        # 計算多變效率（更適合多級分析）
        if self.config.machine_type == MachineType.TURBINE:
            # 渦輪多變效率
            if pressure_ratio < 1.0 and temperature_ratio < 1.0:
                gamma = self.gas_calc.gamma
                ln_pr = math.log(pressure_ratio)
                ln_tr = math.log(temperature_ratio)
                if ln_tr != 0:
                    polytropic_eta = (gamma - 1) / gamma * ln_pr / ln_tr
                else:
                    polytropic_eta = eta
            else:
                polytropic_eta = eta
        else:
            # 壓縮機多變效率
            if pressure_ratio > 1.0 and temperature_ratio > 1.0:
                gamma = self.gas_calc.gamma
                ln_pr = math.log(pressure_ratio)
                ln_tr = math.log(temperature_ratio)
                if ln_pr != 0:
                    polytropic_eta = (gamma - 1) / gamma * ln_tr / ln_pr
                else:
                    polytropic_eta = eta
            else:
                polytropic_eta = eta

        # 保存等熵焓變
        stage.dho = stage.work_output
        stage.dho_is = abs(hoin - outlet_isentropic.ho)

        return StageThermodynamics(
            inlet_state=inlet_state,
            outlet_state=outlet_state,
            outlet_isentropic=outlet_isentropic,
            isentropic_efficiency=eta,
            polytropic_efficiency=polytropic_eta,
            pressure_ratio=pressure_ratio,
            temperature_ratio=temperature_ratio,
            is_choked=is_choked,
        )

    def solve_all_stages(self) -> None:
        """求解所有級。"""
        # 儲存累積狀態用於多級計算
        self._cumulative_states: list[StageThermodynamics] = []

        for i, stage in enumerate(self.config.stages):
            # 如果不是第一級，更新進口條件
            if i > 0 and self._cumulative_states:
                prev_thermo = self._cumulative_states[-1]
                # 更新配置中的進口條件為前一級出口條件
                self._update_inlet_conditions(prev_thermo.outlet_state)

            self.solve_stage(stage)

            # 儲存熱力學狀態
            if hasattr(stage, "thermodynamics"):
                self._cumulative_states.append(stage.thermodynamics)

    def _update_inlet_conditions(self, prev_outlet: FlowState) -> None:
        """更新進口條件（用於多級計算）。

        Args:
            prev_outlet: 前一級出口狀態
        """
        # 更新配置中的進口總壓和總溫
        self.config.gas.poin = prev_outlet.po / 1e5  # Pa -> bar
        self.config.gas.toin = prev_outlet.to

    def calculate_overall_performance(self) -> dict[str, float]:
        """計算整體性能。

        Returns:
            性能參數字典
        """
        total_work = sum(stage.work_output for stage in self.config.stages)
        power = total_work * self.config.mass_flow / 1000.0  # kW

        # 計算整體壓比和溫比（多級疊加）
        overall_pressure_ratio = 1.0
        overall_temperature_ratio = 1.0

        # 收集各級效率和馬赫數
        stage_efficiencies = []
        stage_mach_numbers = []
        choked_stages = []

        for i, stage in enumerate(self.config.stages):
            if hasattr(stage, "thermodynamics") and stage.thermodynamics is not None:
                thermo = stage.thermodynamics
                overall_pressure_ratio *= thermo.pressure_ratio
                overall_temperature_ratio *= thermo.temperature_ratio
                stage_efficiencies.append(thermo.isentropic_efficiency)

                # 記錄進出口馬赫數
                stage_mach_numbers.append(
                    {
                        "stage": i + 1,
                        "inlet_mach": thermo.inlet_state.mach,
                        "outlet_mach": thermo.outlet_state.mach,
                    }
                )

                # 檢查壅塞
                if thermo.is_choked:
                    choked_stages.append(i + 1)

        # 計算整體等熵效率
        from .data_structures import MachineType

        if self.config.machine_type == MachineType.TURBINE:
            # 渦輪整體效率
            if len(stage_efficiencies) > 0:
                # 使用總功和等熵總功計算
                total_dho_is = sum(
                    stage.dho_is for stage in self.config.stages if stage.dho_is > 0
                )
                if total_dho_is > 0:
                    overall_efficiency = total_work / total_dho_is
                else:
                    overall_efficiency = sum(stage_efficiencies) / len(stage_efficiencies)
            else:
                overall_efficiency = 0.9
        else:
            # 壓縮機整體效率
            if len(stage_efficiencies) > 0:
                total_dho_is = sum(
                    stage.dho_is for stage in self.config.stages if stage.dho_is > 0
                )
                if total_work > 0:
                    overall_efficiency = total_dho_is / abs(total_work)
                else:
                    overall_efficiency = sum(stage_efficiencies) / len(stage_efficiencies)
            else:
                overall_efficiency = 0.85

        # 計算整體多變效率
        if overall_pressure_ratio != 1.0 and overall_temperature_ratio != 1.0:
            gamma = self.gas_calc.gamma
            ln_pr = math.log(overall_pressure_ratio)
            ln_tr = math.log(overall_temperature_ratio)

            if self.config.machine_type == MachineType.TURBINE:
                if ln_tr != 0:
                    overall_polytropic_efficiency = (gamma - 1) / gamma * ln_pr / ln_tr
                else:
                    overall_polytropic_efficiency = overall_efficiency
            else:
                if ln_pr != 0:
                    overall_polytropic_efficiency = (gamma - 1) / gamma * ln_tr / ln_pr
                else:
                    overall_polytropic_efficiency = overall_efficiency
        else:
            overall_polytropic_efficiency = overall_efficiency

        # 構建結果字典
        result = {
            "total_work": total_work,  # J/kg
            "power": power,  # kW
            "mass_flow": self.config.mass_flow,  # kg/s
            "stages": self.config.nstages,
            "overall_pressure_ratio": overall_pressure_ratio,
            "overall_temperature_ratio": overall_temperature_ratio,
            "overall_isentropic_efficiency": overall_efficiency,
            "overall_polytropic_efficiency": overall_polytropic_efficiency,
        }

        # 添加壅塞警告
        if choked_stages:
            result["choked_stages"] = choked_stages
            result["choking_warning"] = True
        else:
            result["choking_warning"] = False

        return result

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

    def write_outputs(self, output_dir: str | Path, write_stagen: bool = True) -> None:
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
