# -*- coding: utf-8 -*-
"""MEANGEN 使用示例。

展示如何使用 MEANGEN 進行渦輪/壓縮機平均線設計。
"""

from __future__ import annotations

from pathlib import Path

from multall_turbomachinery_design.meangen.data_structures import (
    FlowType,
    GasProperties,
    InputType,
    MachineType,
    MeangenConfig,
    StageDesign,
)
from multall_turbomachinery_design.meangen.mean_line_solver import MeanLineSolver


def example_turbine_design() -> None:
    """示例：單級軸向渦輪設計。"""
    print("=" * 60)
    print("示例 1：單級軸向渦輪設計")
    print("=" * 60)

    # 創建配置
    config = MeangenConfig(
        machine_type=MachineType.TURBINE,
        flow_type=FlowType.AXIAL,
        gas=GasProperties(
            rgas=287.5,  # 空氣氣體常數 [J/(kg·K)]
            gamma=1.4,  # 比熱比
            poin=1.0,  # 進口總壓 [Bar]
            toin=1200.0,  # 進口總溫 [K]
        ),
        nstages=1,  # 單級
        rpm=10000.0,  # 轉速 [RPM]
        mass_flow=10.0,  # 質量流量 [kg/s]
        design_radius=0.3,  # 設計半徑 [m]
    )

    # 創建級設計（Type A 輸入：使用無量綱參數）
    stage = StageDesign(
        stage_number=1,
        input_type=InputType.TYPE_A,
        phi=0.6,  # 流量係數
        psi=2.0,  # 負荷係數
        reaction=0.5,  # 50% 反應度
        r_design=0.3,  # 設計半徑 [m]
        efficiency=0.90,  # 等熵效率
        axial_chord_1=0.040,  # 轉子軸向弦長 [m]
        axial_chord_2=0.035,  # 定子軸向弦長 [m]
        row_gap=0.020,  # 行間隙 [m]
        stage_gap=0.040,  # 級間隙 [m]
    )
    config.stages.append(stage)

    # 創建求解器並求解
    solver = MeanLineSolver(config)
    performance = solver.run()

    # 打印結果
    print("\n設計參數：")
    print(f"  轉速：{config.rpm:.0f} RPM")
    print(f"  質量流量：{config.mass_flow:.1f} kg/s")
    print(f"  設計半徑：{config.design_radius * 1000:.1f} mm")
    print(f"  流量係數：{stage.phi:.3f}")
    print(f"  負荷係數：{stage.psi:.3f}")
    print(f"  反應度：{stage.reaction:.3f}")

    print("\n速度三角形：")
    if stage.inlet_triangle:
        print("  進口：")
        print(f"    子午速度：{stage.inlet_triangle.vm:.1f} m/s")
        print(f"    切向速度：{stage.inlet_triangle.vtheta:.1f} m/s")
        print(f"    圓周速度：{stage.inlet_triangle.u:.1f} m/s")
        print(f"    絕對角：{stage.inlet_triangle.alpha:.1f}°")
        print(f"    相對角：{stage.inlet_triangle.beta:.1f}°")

    if stage.outlet_triangle:
        print("  出口：")
        print(f"    子午速度：{stage.outlet_triangle.vm:.1f} m/s")
        print(f"    切向速度：{stage.outlet_triangle.vtheta:.1f} m/s")
        print(f"    絕對角：{stage.outlet_triangle.alpha:.1f}°")
        print(f"    相對角：{stage.outlet_triangle.beta:.1f}°")

    print("\n葉片排：")
    if stage.rotor:
        print("  轉子：")
        print(f"    葉片數：{stage.rotor.n_blades}")
        print(f"    軸向弦長：{stage.rotor.axial_chord * 1000:.1f} mm")
    if stage.stator:
        print("  定子：")
        print(f"    葉片數：{stage.stator.n_blades}")
        print(f"    軸向弦長：{stage.stator.axial_chord * 1000:.1f} mm")

    print("\n性能：")
    print(f"  比功：{abs(performance['total_work']):.0f} J/kg")
    print(f"  功率：{abs(performance['power']):.2f} kW")

    print()


def example_compressor_design() -> None:
    """示例：單級軸向壓縮機設計。"""
    print("=" * 60)
    print("示例 2：單級軸向壓縮機設計")
    print("=" * 60)

    # 創建配置
    config = MeangenConfig(
        machine_type=MachineType.COMPRESSOR,
        flow_type=FlowType.AXIAL,
        gas=GasProperties(
            rgas=287.5,
            gamma=1.4,
            poin=1.0,
            toin=288.15,  # 15°C
        ),
        nstages=1,
        rpm=15000.0,
        mass_flow=20.0,
        design_radius=0.25,
    )

    # 創建級設計
    stage = StageDesign(
        stage_number=1,
        input_type=InputType.TYPE_A,
        phi=0.5,  # 壓縮機通常較低的流量係數
        psi=0.4,  # 壓縮機負荷係數較低
        reaction=0.5,
        r_design=0.25,
        efficiency=0.88,
        axial_chord_1=0.030,
        axial_chord_2=0.028,
        row_gap=0.015,
        stage_gap=0.030,
    )
    config.stages.append(stage)

    # 求解
    solver = MeanLineSolver(config)
    performance = solver.run()

    # 打印結果
    print("\n設計參數：")
    print(f"  轉速：{config.rpm:.0f} RPM")
    print(f"  質量流量：{config.mass_flow:.1f} kg/s")
    print(f"  設計半徑：{config.design_radius * 1000:.1f} mm")

    print("\n性能：")
    print(f"  比功：{abs(performance['total_work']):.0f} J/kg (輸入)")
    print(f"  功率：{abs(performance['power']):.2f} kW")

    if stage.rotor:
        print(f"\n轉子葉片數：{stage.rotor.n_blades}")
    if stage.stator:
        print(f"定子葉片數：{stage.stator.n_blades}")

    print()


def example_multistage_turbine() -> None:
    """示例：三級軸向渦輪設計。"""
    print("=" * 60)
    print("示例 3：三級軸向渦輪設計")
    print("=" * 60)

    config = MeangenConfig(
        machine_type=MachineType.TURBINE,
        flow_type=FlowType.AXIAL,
        gas=GasProperties(rgas=287.5, gamma=1.4, poin=5.0, toin=1400.0),
        nstages=3,
        rpm=8000.0,
        mass_flow=15.0,
        design_radius=0.35,
    )

    # 三級設計，負荷遞減
    loadings = [2.2, 2.0, 1.8]
    for i in range(3):
        stage = StageDesign(
            stage_number=i + 1,
            input_type=InputType.TYPE_A,
            phi=0.6,
            psi=loadings[i],
            reaction=0.5,
            r_design=0.35,
            efficiency=0.91,
            axial_chord_1=0.045,
            axial_chord_2=0.040,
            row_gap=0.022,
            stage_gap=0.045,
        )
        config.stages.append(stage)

    # 求解
    solver = MeanLineSolver(config)
    performance = solver.run()

    # 打印結果
    print("\n整體性能：")
    print(f"  總比功：{abs(performance['total_work']):.0f} J/kg")
    print(f"  總功率：{abs(performance['power']):.2f} kW")
    print(f"  級數：{performance['stages']}")

    print("\n各級資訊：")
    for i, stage in enumerate(config.stages, 1):
        print(f"  級 {i}：")
        print(f"    負荷係數：{stage.psi:.2f}")
        if stage.rotor:
            print(f"    轉子葉片數：{stage.rotor.n_blades}")
        if stage.stator:
            print(f"    定子葉片數：{stage.stator.n_blades}")

    print()


def example_write_output_files() -> None:
    """示例：寫入輸出檔案。"""
    print("=" * 60)
    print("示例 4：寫入輸出檔案")
    print("=" * 60)

    # 創建簡單配置
    config = MeangenConfig(
        machine_type=MachineType.TURBINE,
        flow_type=FlowType.AXIAL,
        gas=GasProperties(rgas=287.5, gamma=1.4, poin=1.0, toin=1200.0),
        nstages=1,
        rpm=10000.0,
        mass_flow=10.0,
        design_radius=0.3,
    )

    stage = StageDesign(
        stage_number=1,
        input_type=InputType.TYPE_A,
        phi=0.6,
        psi=2.0,
        reaction=0.5,
        r_design=0.3,
        efficiency=0.90,
    )
    config.stages.append(stage)

    # 求解並寫入檔案
    solver = MeanLineSolver(config)
    output_dir = Path("output")
    _performance = solver.run(output_dir)  # 寫入輸出檔案

    print(f"\n輸出檔案已寫入：{output_dir.absolute()}")
    print("  - meangen.out：平均線設計結果")
    print("  - stagen.dat：STAGEN 輸入檔案")

    print()


if __name__ == "__main__":
    print("\nMULTALL MEANGEN - 渦輪機械平均線設計工具")
    print("=" * 60)
    print()

    # 運行所有示例
    example_turbine_design()
    example_compressor_design()
    example_multistage_turbine()
    example_write_output_files()

    print("所有示例完成！")
    print()
