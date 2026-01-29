# -*- coding: utf-8 -*-
"""命令行主介面。

提供 MULTALL 系統的主命令行入口。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def create_parser() -> argparse.ArgumentParser:
    """創建命令行解析器。

    Returns:
        argparse 解析器
    """
    from multall_turbomachinery_design import __version__

    parser = argparse.ArgumentParser(
        prog="multall",
        description="MULTALL 渦輪機械設計系統 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  multall info                          顯示系統資訊
  multall meangen input.json -o output  運行平均線設計
  multall stagen input.dat -o output    生成葉片幾何
  multall plot convergence data.csv     繪製收斂歷史
  multall gui                           啟動圖形介面

更多資訊請參閱: https://github.com/LostSunset/multall-turbomachinery-design
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="安靜模式，減少輸出",
    )

    # 子命令
    subparsers = parser.add_subparsers(
        title="命令",
        dest="command",
        help="可用命令",
    )

    # info 命令
    _add_info_parser(subparsers)

    # meangen 命令
    _add_meangen_parser(subparsers)

    # stagen 命令
    _add_stagen_parser(subparsers)

    # plot 命令
    _add_plot_parser(subparsers)

    # gui 命令
    _add_gui_parser(subparsers)

    # validate 命令
    _add_validate_parser(subparsers)

    return parser


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 info 子命令。"""
    info_parser = subparsers.add_parser(
        "info",
        help="顯示系統資訊",
        description="顯示 MULTALL 系統的版本和配置資訊",
    )
    info_parser.add_argument(
        "--check-deps",
        action="store_true",
        help="檢查依賴項",
    )


def _add_meangen_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 meangen 子命令。"""
    meangen_parser = subparsers.add_parser(
        "meangen",
        help="運行平均線設計",
        description="執行一維平均線設計計算",
    )
    meangen_parser.add_argument(
        "input",
        type=Path,
        help="輸入配置文件 (JSON 或 meangen.in 格式)",
    )
    meangen_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="輸出目錄 (預設: output)",
    )
    meangen_parser.add_argument(
        "--format",
        choices=["json", "fortran"],
        default="json",
        help="輸入文件格式 (預設: json)",
    )


def _add_stagen_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 stagen 子命令。"""
    stagen_parser = subparsers.add_parser(
        "stagen",
        help="生成葉片幾何",
        description="生成 3D 葉片幾何和網格",
    )
    stagen_parser.add_argument(
        "input",
        type=Path,
        help="輸入配置文件 (JSON 或 stagen.dat 格式)",
    )
    stagen_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="輸出目錄 (預設: output)",
    )
    stagen_parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="僅生成網格，不輸出幾何文件",
    )


def _add_plot_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 plot 子命令。"""
    plot_parser = subparsers.add_parser(
        "plot",
        help="視覺化結果",
        description="繪製計算結果圖表",
    )

    plot_subparsers = plot_parser.add_subparsers(
        title="圖表類型",
        dest="plot_type",
        help="可用圖表類型",
    )

    # 收斂歷史
    conv_parser = plot_subparsers.add_parser(
        "convergence",
        help="繪製收斂歷史",
    )
    conv_parser.add_argument("data", type=Path, help="數據文件 (CSV)")
    conv_parser.add_argument("-o", "--output", type=Path, help="輸出圖片路徑")
    conv_parser.add_argument("--no-show", action="store_true", help="不顯示圖形")

    # 速度三角形
    tri_parser = plot_subparsers.add_parser(
        "triangle",
        help="繪製速度三角形",
    )
    tri_parser.add_argument("data", type=Path, help="數據文件 (JSON)")
    tri_parser.add_argument("-o", "--output", type=Path, help="輸出圖片路徑")
    tri_parser.add_argument("--no-show", action="store_true", help="不顯示圖形")

    # 葉片截面
    blade_parser = plot_subparsers.add_parser(
        "blade",
        help="繪製葉片截面",
    )
    blade_parser.add_argument("data", type=Path, help="數據文件 (CSV 或 JSON)")
    blade_parser.add_argument("-o", "--output", type=Path, help="輸出圖片路徑")
    blade_parser.add_argument("--no-show", action="store_true", help="不顯示圖形")
    blade_parser.add_argument("--cascade", action="store_true", help="繪製葉柵視圖")


def _add_gui_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 gui 子命令。"""
    subparsers.add_parser(
        "gui",
        help="啟動圖形介面",
        description="啟動 MULTALL 圖形使用者介面",
    )


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    """添加 validate 子命令。"""
    validate_parser = subparsers.add_parser(
        "validate",
        help="驗證輸入文件",
        description="驗證輸入配置文件的格式和內容",
    )
    validate_parser.add_argument(
        "input",
        type=Path,
        help="輸入配置文件",
    )
    validate_parser.add_argument(
        "--type",
        choices=["meangen", "stagen", "multall"],
        help="文件類型（自動檢測如不指定）",
    )


def run_info(args: Namespace) -> int:
    """執行 info 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    from multall_turbomachinery_design import __version__

    print("=" * 50)
    print("MULTALL 渦輪機械設計系統")
    print("=" * 50)
    print(f"版本: {__version__}")
    print(f"Python: {sys.version}")
    print()

    if args.check_deps:
        print("依賴項檢查:")
        _check_dependency("numpy", "數值計算")
        _check_dependency("matplotlib", "視覺化")
        _check_dependency("PySide6", "圖形介面")
        print()

    print("可用模組:")
    print("  - MEANGEN: 一維平均線設計")
    print("  - STAGEN:  葉片幾何生成")
    print("  - MULTALL: 3D Navier-Stokes 求解器")
    print()
    print("使用 'multall --help' 查看可用命令")

    return 0


def _check_dependency(name: str, description: str) -> None:
    """檢查依賴項。"""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "未知")
        print(f"  ✓ {name} ({version}) - {description}")
    except ImportError:
        print(f"  ✗ {name} - {description} [未安裝]")


def run_meangen(args: Namespace) -> int:
    """執行 meangen 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    import json

    from multall_turbomachinery_design.meangen import (
        FlowType,
        GasProperties,
        InputType,
        MachineType,
        MeangenConfig,
        MeanLineSolver,
        StageDesign,
    )

    input_path = args.input
    output_dir = args.output

    if not input_path.exists():
        print(f"錯誤: 輸入文件不存在: {input_path}")
        return 1

    print(f"讀取配置: {input_path}")

    try:
        # 讀取 JSON 配置
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        # 創建配置
        machine_type = MachineType[data.get("machine_type", "TURBINE").upper()]
        flow_type = FlowType[data.get("flow_type", "AXIAL").upper()]

        gas_data = data.get("gas", {})
        gas = GasProperties(
            gamma=gas_data.get("gamma", 1.4),
            rgas=gas_data.get("rgas", 287.05),
            poin=gas_data.get("poin", 1.0),
            toin=gas_data.get("toin", 1200.0),
        )

        config = MeangenConfig(
            machine_type=machine_type,
            flow_type=flow_type,
            gas=gas,
            nstages=data.get("nstages", 1),
            rpm=data.get("rpm", 10000.0),
            mass_flow=data.get("mass_flow", 10.0),
            design_radius=data.get("design_radius", 0.3),
        )

        # 添加級設計
        for stage_data in data.get("stages", []):
            input_type = InputType[stage_data.get("input_type", "TYPE_A").upper()]
            stage = StageDesign(
                stage_number=stage_data.get("stage_number", 1),
                input_type=input_type,
                phi=stage_data.get("phi", 0.6),
                psi=stage_data.get("psi", 2.0),
                reaction=stage_data.get("reaction", 0.5),
                r_design=stage_data.get("r_design", config.design_radius),
                efficiency=stage_data.get("efficiency", 0.90),
            )
            config.stages.append(stage)

        # 如果沒有定義級，添加預設級
        if not config.stages:
            config.stages.append(
                StageDesign(
                    stage_number=1,
                    input_type=InputType.TYPE_A,
                    phi=0.6,
                    psi=2.0,
                    reaction=0.5,
                    r_design=config.design_radius,
                    efficiency=0.90,
                )
            )

        print(f"機器類型: {machine_type.name}")
        print(f"流動類型: {flow_type.name}")
        print(f"級數: {config.nstages}")

        # 運行求解器
        solver = MeanLineSolver(config)
        result = solver.run()

        # 創建輸出目錄
        output_dir.mkdir(parents=True, exist_ok=True)

        # 輸出結果
        solver.write_output(output_dir)

        print()
        print("計算完成!")
        print(f"功率: {abs(result.get('power', 0)):.2f} W")
        print(f"壓比: {result.get('pressure_ratio', 1.0):.4f}")
        print(f"效率: {result.get('efficiency', 0.0):.4f}")
        print(f"輸出目錄: {output_dir}")

        return 0

    except Exception as e:
        print(f"錯誤: {e}")
        return 1


def run_stagen(args: Namespace) -> int:
    """執行 stagen 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    print(f"讀取配置: {args.input}")
    print("STAGEN 命令行介面開發中...")
    print(f"輸出目錄: {args.output}")

    # TODO: 實作 STAGEN CLI
    return 0


def run_plot(args: Namespace) -> int:
    """執行 plot 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    import json

    import numpy as np

    plot_type = args.plot_type

    if plot_type is None:
        print("錯誤: 請指定圖表類型")
        print("使用 'multall plot --help' 查看可用類型")
        return 1

    data_path = args.data
    if not data_path.exists():
        print(f"錯誤: 數據文件不存在: {data_path}")
        return 1

    print(f"讀取數據: {data_path}")

    try:
        if plot_type == "convergence":
            from multall_turbomachinery_design.visualization import (
                plot_convergence_history,
            )

            # 讀取 CSV
            data = np.genfromtxt(data_path, delimiter=",", names=True)
            iterations = (
                data["iteration"] if "iteration" in data.dtype.names else np.arange(len(data))
            )
            residual = (
                data["residual"] if "residual" in data.dtype.names else data[data.dtype.names[0]]
            )

            efficiency = data["efficiency"] if "efficiency" in data.dtype.names else None
            mass_flow = data["mass_flow"] if "mass_flow" in data.dtype.names else None

            plot_convergence_history(
                iterations,
                residual,
                efficiency=efficiency,
                mass_flow=mass_flow,
                show=not args.no_show,
                save_path=str(args.output) if args.output else None,
            )

        elif plot_type == "triangle":
            from multall_turbomachinery_design.visualization import (
                plot_velocity_triangle,
            )

            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)

            plot_velocity_triangle(
                data["inlet"],
                data["outlet"],
                title=data.get("title", "速度三角形"),
                show=not args.no_show,
                save_path=str(args.output) if args.output else None,
            )

        elif plot_type == "blade":
            from multall_turbomachinery_design.visualization import plot_blade_profile

            # 讀取 CSV 或 JSON
            if data_path.suffix == ".json":
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
                x_upper = np.array(data["x_upper"])
                y_upper = np.array(data["y_upper"])
                x_lower = np.array(data["x_lower"])
                y_lower = np.array(data["y_lower"])
            else:
                data = np.genfromtxt(data_path, delimiter=",", names=True)
                x_upper = data["x_upper"]
                y_upper = data["y_upper"]
                x_lower = data["x_lower"]
                y_lower = data["y_lower"]

            plot_blade_profile(
                x_upper,
                y_upper,
                x_lower,
                y_lower,
                show=not args.no_show,
                save_path=str(args.output) if args.output else None,
            )

        else:
            print(f"錯誤: 未知的圖表類型: {plot_type}")
            return 1

        if args.output:
            print(f"圖片已儲存: {args.output}")

        return 0

    except Exception as e:
        print(f"錯誤: {e}")
        return 1


def run_gui(args: Namespace) -> int:
    """執行 gui 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    try:
        from multall_turbomachinery_design.ui import main

        return main()
    except ImportError as e:
        print(f"錯誤: 無法啟動圖形介面: {e}")
        print("請確保已安裝 PySide6")
        return 1


def run_validate(args: Namespace) -> int:
    """執行 validate 命令。

    Args:
        args: 命令行參數

    Returns:
        退出碼
    """
    import json

    input_path = args.input

    if not input_path.exists():
        print(f"錯誤: 文件不存在: {input_path}")
        return 1

    print(f"驗證文件: {input_path}")

    try:
        # 嘗試讀取 JSON
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        print("✓ JSON 格式正確")

        # 檢查必要欄位
        errors = []
        warnings = []

        if "machine_type" not in data:
            warnings.append("缺少 'machine_type'，將使用預設值 'TURBINE'")

        if "flow_type" not in data:
            warnings.append("缺少 'flow_type'，將使用預設值 'AXIAL'")

        if "stages" not in data or len(data.get("stages", [])) == 0:
            warnings.append("缺少 'stages'，將使用預設級設計")

        # 輸出結果
        if errors:
            print("\n錯誤:")
            for err in errors:
                print(f"  ✗ {err}")
            return 1

        if warnings:
            print("\n警告:")
            for warn in warnings:
                print(f"  ! {warn}")

        print("\n✓ 驗證通過")
        return 0

    except json.JSONDecodeError as e:
        print(f"✗ JSON 格式錯誤: {e}")
        return 1
    except Exception as e:
        print(f"✗ 驗證失敗: {e}")
        return 1


def app(args: list[str] | None = None) -> int:
    """命令行應用程式入口。

    Args:
        args: 命令行參數（None 表示使用 sys.argv）

    Returns:
        退出碼
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # 執行對應命令
    commands = {
        "info": run_info,
        "meangen": run_meangen,
        "stagen": run_stagen,
        "plot": run_plot,
        "gui": run_gui,
        "validate": run_validate,
    }

    handler = commands.get(parsed_args.command)
    if handler:
        return handler(parsed_args)
    else:
        print(f"錯誤: 未知命令 '{parsed_args.command}'")
        return 1


def main() -> None:
    """主函數。"""
    sys.exit(app())


if __name__ == "__main__":
    main()
