# -*- coding: utf-8 -*-
"""MULTALL 使用示例。

展示如何使用 MULTALL 3D Navier-Stokes 求解器進行渦輪機械流場分析。

示例包括：
1. 簡單渦輪通道流場初始化
2. 求解器設置和運行
3. 後處理和結果分析
4. 混合平面模型使用
5. 逆向設計演示
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multall_turbomachinery_design.multall import (
    ConvergenceMonitor,
    # 數據結構
    FlowField,
    FlowFieldExtractor,
    # 求解器和計算器
    FluxCalculator,
    GasProperties,
    GasType,
    Grid3D,
    InverseDesignParameters,
    # 逆向設計
    InverseDesignSolver,
    InverseDesignType,
    # 黏性模型
    MixingLengthModel,
    # 混合平面
    MixingPlaneModel,
    MixingPlaneParameters,
    MixingPlaneType,
    PerformanceCalculator,
    ResultExporter,
    SolverParameters,
    TimeStepper,
    WallDistanceCalculator,
)


def example_flow_field_initialization() -> None:
    """示例 1：流場初始化。

    展示如何創建和初始化 3D 流場。
    """
    print("=" * 60)
    print("示例 1：流場初始化")
    print("=" * 60)

    # 定義網格尺寸
    im = 17  # 周向網格點數
    jm = 50  # 軸向網格點數
    km = 9  # 跨向網格點數

    # 創建流場
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()  # 分配數組內存

    print(f"網格尺寸: {im} x {jm} x {km}")
    print(f"總網格點數: {im * jm * km}")

    # 設置初始條件（均勻流場）
    # 入口條件
    p_in = 200000.0  # 入口壓力 [Pa]
    t_in = 400.0  # 入口靜溫 [K]
    vx_in = 100.0  # 入口軸向速度 [m/s]

    # 氣體性質
    gamma = 1.4
    rgas = 287.0
    rho_in = p_in / (rgas * t_in)

    # 初始化整個流場
    flow.rho[:] = rho_in
    flow.vx[:] = vx_in
    flow.vr[:] = 0.0
    flow.vt[:] = 0.0
    flow.p[:] = p_in
    flow.t_static[:] = t_in

    print("\n初始條件:")
    print(f"  壓力: {p_in:.0f} Pa")
    print(f"  溫度: {t_in:.1f} K")
    print(f"  密度: {rho_in:.3f} kg/m³")
    print(f"  軸向速度: {vx_in:.1f} m/s")

    # 計算馬赫數
    a = np.sqrt(gamma * rgas * t_in)
    mach = vx_in / a
    print(f"  音速: {a:.1f} m/s")
    print(f"  馬赫數: {mach:.3f}")

    print("\n流場初始化完成!")


def example_gas_properties() -> None:
    """示例 2：氣體性質計算。

    展示如何使用 GasCalculator 進行熱力學計算。
    """
    print("\n" + "=" * 60)
    print("示例 2：氣體性質計算")
    print("=" * 60)

    # 創建空氣計算器
    from multall_turbomachinery_design.multall import create_air_calculator

    gas_calc = create_air_calculator()

    # 測試條件
    t_static = 400.0  # 靜溫 [K]
    v_mag = 200.0  # 速度 [m/s]

    # 計算焓
    h_static = gas_calc.enthalpy_from_temperature(t_static)
    h_total = h_static + 0.5 * v_mag**2

    # 從總焓計算總溫
    t_total = gas_calc.temperature_from_enthalpy(h_total)

    print("\n熱力學計算:")
    print(f"  靜溫: {t_static:.1f} K")
    print(f"  速度: {v_mag:.1f} m/s")
    print(f"  靜焓: {h_static / 1000:.2f} kJ/kg")
    print(f"  總焓: {h_total / 1000:.2f} kJ/kg")
    print(f"  總溫: {t_total:.1f} K")

    # 等熵關係計算（使用基本公式）
    p_static = 100000.0  # 靜壓 [Pa]
    gamma = gas_calc.gas.gamma
    p_total = p_static * (t_total / t_static) ** (gamma / (gamma - 1))

    print("\n等熵關係:")
    print(f"  靜壓: {p_static:.0f} Pa")
    print(f"  總壓: {p_total:.0f} Pa")
    print(f"  壓比: {p_total / p_static:.3f}")

    # 馬赫數等熵關係
    mach = 0.5
    t_ratio, p_ratio, rho_ratio, a_ratio = gas_calc.isentropic_relations(mach)
    print(f"\n馬赫數 {mach} 的等熵關係:")
    print(f"  T/T0: {t_ratio:.4f}")
    print(f"  p/p0: {p_ratio:.4f}")
    print(f"  ρ/ρ0: {rho_ratio:.4f}")


def example_flux_calculation() -> None:
    """示例 3：通量計算。

    展示通量計算器的基本概念。
    """
    print("\n" + "=" * 60)
    print("示例 3：通量計算（Roe 格式）")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建通量計算器（展示如何創建）
    flux_calc = FluxCalculator(gas)  # noqa: F841

    print(f"\n通量計算器已創建 (類型: {type(flux_calc).__name__})")
    print(f"  氣體比熱比: {gas.gamma}")
    print(f"  定壓比熱: {gas.cp} J/(kg·K)")

    # 創建簡單流場展示基本數據結構
    im, jm, km = 5, 10, 5
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()

    # 設置流場（帶壓力梯度）
    for j in range(jm):
        frac = j / (jm - 1)
        flow.rho[:, j, :] = 1.5 - 0.3 * frac
        flow.vx[:, j, :] = 100.0 + 50.0 * frac
        flow.vr[:, j, :] = 0.0
        flow.vt[:, j, :] = 0.0
        flow.p[:, j, :] = 200000.0 - 100000.0 * frac
        flow.t_static[:, j, :] = 400.0 - 50.0 * frac

    print("\n流場設置:")
    print(f"  網格尺寸: {im} x {jm} x {km}")
    print(f"  入口密度: {flow.rho[0, 0, 0]:.3f} kg/m³")
    print(f"  出口密度: {flow.rho[0, -1, 0]:.3f} kg/m³")
    print(f"  入口速度: {flow.vx[0, 0, 0]:.1f} m/s")
    print(f"  出口速度: {flow.vx[0, -1, 0]:.1f} m/s")
    print(f"  入口壓力: {flow.p[0, 0, 0]:.0f} Pa")
    print(f"  出口壓力: {flow.p[0, -1, 0]:.0f} Pa")

    # 計算質量通量（簡化示例）
    mass_flux_in = flow.rho[:, 0, :] * flow.vx[:, 0, :]
    mass_flux_out = flow.rho[:, -1, :] * flow.vx[:, -1, :]
    print("\n質量通量:")
    print(f"  入口平均質量通量: {mass_flux_in.mean():.2f} kg/(m²·s)")
    print(f"  出口平均質量通量: {mass_flux_out.mean():.2f} kg/(m²·s)")


def example_time_stepping() -> None:
    """示例 4：時間步進。

    展示如何使用不同的時間推進方法。
    """
    print("\n" + "=" * 60)
    print("示例 4：時間步進方法")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建求解器參數
    solver_params = SolverParameters(
        cfl=0.5,
        max_steps=5000,
        convergence_limit=1e-5,
    )

    # 創建時間步進器（展示如何創建）
    time_stepper = TimeStepper(gas, solver_params)  # noqa: F841

    print(f"\n時間步進器已創建 (類型: {type(time_stepper).__name__})")
    print(f"  CFL 數: {solver_params.cfl}")
    print(f"  最大步數: {solver_params.max_steps}")
    print(f"  收斂準則: {solver_params.convergence_limit}")

    # 創建收斂監測器
    monitor = ConvergenceMonitor(
        convergence_limit=1e-5,
        history_size=100,
    )

    print("\n收斂監測器設置:")
    print(f"  收斂準則: {monitor.convergence_limit}")
    print(f"  歷史大小: {monitor.history_size}")

    # 模擬收斂過程
    print("\n模擬收斂過程:")
    residuals = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    for i, res in enumerate(residuals):
        monitor.add_residual(res)
        print(f"  步數 {i + 1}: 殘差 = {res:.2e}, 收斂 = {monitor.is_converged()}")


def example_viscous_model() -> None:
    """示例 5：黏性模型。

    展示如何使用混合長度湍流模型。
    """
    print("\n" + "=" * 60)
    print("示例 5：黏性模型（混合長度）")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建混合長度模型（展示如何創建）
    mixing_length = MixingLengthModel(gas)  # noqa: F841

    print(f"\n混合長度模型已創建 (類型: {type(mixing_length).__name__})")
    print(f"  von Kármán 常數: {MixingLengthModel.KAPPA}")
    print(f"  van Driest 常數: {MixingLengthModel.A_PLUS}")

    # 創建簡單網格演示壁面距離計算
    im, jm, km = 5, 20, 9
    grid = Grid3D(im=im, jm=jm, km=km)
    grid.initialize()

    # 設置網格半徑（模擬環形通道）
    for j in range(jm):
        for k in range(km):
            grid.r[j, k] = 0.3 + 0.05 * k / (km - 1)  # 內徑 0.3m，外徑 0.35m

    # 創建壁面距離計算器（展示如何創建）
    wall_calc = WallDistanceCalculator(grid)  # noqa: F841

    print(f"\n壁面距離計算器已創建 (類型: {type(wall_calc).__name__})")
    print(f"  網格: {im} x {jm} x {km}")
    print(f"  內徑: {grid.r[0, 0]:.3f} m")
    print(f"  外徑: {grid.r[0, -1]:.3f} m")
    print(f"  通道高度: {grid.r[0, -1] - grid.r[0, 0]:.3f} m")


def example_mixing_plane() -> None:
    """示例 6：混合平面模型。

    展示如何使用混合平面處理多排葉片交界面。
    """
    print("\n" + "=" * 60)
    print("示例 6：混合平面模型")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建混合平面參數
    params = MixingPlaneParameters(enabled=True, rfmix=0.025)

    # 創建混合平面模型
    mixing_plane = MixingPlaneModel(gas, params)

    # 設置平均類型
    mixing_plane.averaging_type = MixingPlaneType.CIRCUMFERENTIAL_AVERAGE

    print("\n混合平面模型設置:")
    print(f"  啟用: {params.enabled}")
    print(f"  鬆弛因子: {params.rfmix}")
    print(f"  平均類型: {mixing_plane.averaging_type.name}")

    # 添加交界面
    interface = mixing_plane.add_interface(j_upstream=25, j_downstream=26)

    print("\n交界面設置:")
    print(f"  上游 J 索引: {interface.j_upstream}")
    print(f"  下游 J 索引: {interface.j_downstream}")

    # 創建測試流場
    im, jm, km = 9, 50, 9
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()

    # 設置帶周向變化的流場
    for i in range(im):
        theta = 2 * np.pi * i / im
        for j in range(jm):
            for k in range(km):
                flow.rho[i, j, k] = 1.2 + 0.1 * np.sin(theta)
                flow.vx[i, j, k] = 100.0 + 10.0 * np.cos(theta)
                flow.vr[i, j, k] = 0.0
                flow.vt[i, j, k] = 50.0 + 5.0 * np.sin(theta)
                flow.p[i, j, k] = 101325.0 + 1000.0 * np.sin(theta)
                flow.t_static[i, j, k] = 300.0

    # 計算周向平均
    avg = mixing_plane.compute_circumferential_average(flow, j_index=25)

    print("\n周向平均結果（J=25）:")
    print(f"  平均密度: {avg['rho'].mean():.4f} kg/m³")
    print(f"  平均軸向速度: {avg['vx'].mean():.2f} m/s")
    print(f"  平均切向速度: {avg['vt'].mean():.2f} m/s")
    print(f"  平均壓力: {avg['p'].mean():.0f} Pa")


def example_performance_calculation() -> None:
    """示例 7：性能計算。

    展示如何使用後處理工具計算渦輪性能。
    """
    print("\n" + "=" * 60)
    print("示例 7：性能計算")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建性能計算器
    perf_calc = PerformanceCalculator(gas)

    # 創建模擬渦輪流場
    im, jm, km = 5, 30, 9
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()

    # 入口條件（高壓高溫）
    flow.rho[:, 0:5, :] = 2.5
    flow.vx[:, 0:5, :] = 80.0
    flow.vr[:, 0:5, :] = 0.0
    flow.vt[:, 0:5, :] = 0.0
    flow.p[:, 0:5, :] = 400000.0
    flow.t_static[:, 0:5, :] = 500.0

    # 出口條件（低壓低溫，膨脹後）
    flow.rho[:, 25:, :] = 1.2
    flow.vx[:, 25:, :] = 150.0
    flow.vr[:, 25:, :] = 0.0
    flow.vt[:, 25:, :] = -100.0
    flow.p[:, 25:, :] = 100000.0
    flow.t_static[:, 25:, :] = 350.0

    # 中間區域線性過渡
    for j in range(5, 25):
        frac = (j - 5) / 20.0
        flow.rho[:, j, :] = 2.5 - 1.3 * frac
        flow.vx[:, j, :] = 80.0 + 70.0 * frac
        flow.vt[:, j, :] = -100.0 * frac
        flow.p[:, j, :] = 400000.0 - 300000.0 * frac
        flow.t_static[:, j, :] = 500.0 - 150.0 * frac

    # 計算整機性能
    metrics = perf_calc.compute_overall_performance(flow)

    print("\n整機性能:")
    print(f"  質量流量: {metrics.mass_flow:.2f} kg/s")
    print(f"  功率輸出: {metrics.power_output:.2f} kW")
    print(f"  總-總效率: {metrics.total_to_total_efficiency:.4f}")
    print(f"  總-靜效率: {metrics.total_to_static_efficiency:.4f}")
    print(f"  壓比: {metrics.pressure_ratio:.3f}")
    print(f"  溫比: {metrics.temperature_ratio:.3f}")

    print("\n入口條件:")
    print(f"  總壓: {metrics.inlet_total_pressure:.0f} Pa")
    print(f"  總溫: {metrics.inlet_total_temperature:.1f} K")
    print(f"  馬赫數: {metrics.inlet_mach:.3f}")

    print("\n出口條件:")
    print(f"  總壓: {metrics.exit_total_pressure:.0f} Pa")
    print(f"  靜壓: {metrics.exit_static_pressure:.0f} Pa")
    print(f"  總溫: {metrics.exit_total_temperature:.1f} K")
    print(f"  馬赫數: {metrics.exit_mach:.3f}")


def example_result_export() -> None:
    """示例 8：結果導出。

    展示如何導出計算結果。
    """
    print("\n" + "=" * 60)
    print("示例 8：結果導出")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建結果導出器
    exporter = ResultExporter(gas)

    # 創建示例流場
    im, jm, km = 5, 20, 5
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()
    flow.rho[:] = 1.2
    flow.vx[:] = 100.0
    flow.vr[:] = 0.0
    flow.vt[:] = 50.0
    flow.p[:] = 101325.0
    flow.t_static[:] = 300.0

    # 創建輸出目錄
    output_dir = Path("output/multall_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 導出收斂歷史
    residuals = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    exporter.export_convergence_history(residuals, output_dir / "convergence.csv")
    print(f"\n收斂歷史已導出至: {output_dir / 'convergence.csv'}")

    # 創建性能指標並導出
    from multall_turbomachinery_design.multall.postprocessing import PerformanceMetrics

    metrics = PerformanceMetrics(
        mass_flow=10.0,
        power_output=500.0,
        total_to_total_efficiency=0.90,
        total_to_static_efficiency=0.85,
        pressure_ratio=4.0,
        temperature_ratio=1.3,
        inlet_total_pressure=400000.0,
        inlet_total_temperature=500.0,
        inlet_mach=0.3,
        exit_total_pressure=100000.0,
        exit_static_pressure=90000.0,
        exit_total_temperature=385.0,
        exit_mach=0.5,
    )

    exporter.export_performance_summary(metrics, output_dir / "performance.txt")
    print(f"性能摘要已導出至: {output_dir / 'performance.txt'}")

    # 提取並導出站點數據
    extractor = FlowFieldExtractor(gas)
    station_data = extractor.extract_at_j_station(flow, None, j_index=10)
    exporter.export_station_data_csv(station_data, output_dir / "station_j10.csv")
    print(f"站點數據已導出至: {output_dir / 'station_j10.csv'}")

    print(f"\n所有結果已導出至: {output_dir}")


def example_inverse_design() -> None:
    """示例 9：逆向設計。

    展示如何使用逆向設計功能。
    """
    print("\n" + "=" * 60)
    print("示例 9：逆向設計")
    print("=" * 60)

    # 創建氣體性質
    gas = GasProperties(gamma=1.4, cp=1005.0, gas_type=GasType.PERFECT_GAS)

    # 創建逆向設計參數
    params = InverseDesignParameters(
        enabled=True,
        design_type=InverseDesignType.EXIT_ANGLE,
        target_exit_angle=np.radians(-60.0),  # 目標出口角度
        j_leading_edge=5,
        j_trailing_edge=15,
        rotation_relaxation=0.5,
    )

    # 創建逆向設計求解器
    inverse_solver = InverseDesignSolver(gas, params)

    print("\n逆向設計設置:")
    print(f"  設計類型: {params.design_type.name}")
    print(f"  目標出口角度: {np.degrees(params.target_exit_angle):.1f}°")
    print(f"  旋轉鬆弛因子: {params.rotation_relaxation}")

    # 創建測試流場
    im, jm, km = 5, 20, 9
    flow = FlowField(im=im, jm=jm, km=km)
    flow.initialize()
    flow.rho[:] = 1.2
    flow.vx[:] = 100.0
    flow.vr[:] = 0.0
    flow.vt[:] = -50.0
    flow.p[:] = 101325.0
    flow.t_static[:] = 300.0

    # 執行逆向設計迭代
    print("\n執行逆向設計迭代:")
    for i in range(3):
        result = inverse_solver.iterate(flow, omega=1000.0, n_blades=50)
        print(
            f"  迭代 {result.iterations}: "
            f"當前角度 = {np.degrees(result.current_exit_angle):.2f}°, "
            f"旋轉修正 = {np.degrees(result.rotation_angle):.3f}°"
        )

    # 獲取收斂歷史
    history = inverse_solver.get_convergence_history()
    print(f"\n收斂歷史記錄: {len(history['iterations'])} 次迭代")


def main() -> None:
    """主函數：運行所有示例。"""
    print("\n" + "=" * 60)
    print("       MULTALL 3D Navier-Stokes 求解器使用示例")
    print("=" * 60)

    try:
        # 運行所有示例
        example_flow_field_initialization()
        example_gas_properties()
        example_flux_calculation()
        example_time_stepping()
        example_viscous_model()
        example_mixing_plane()
        example_performance_calculation()
        example_result_export()
        example_inverse_design()

        print("\n" + "=" * 60)
        print("所有示例運行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
