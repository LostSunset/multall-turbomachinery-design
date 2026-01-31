# -*- coding: utf-8 -*-
"""MEANGEN 面板。

提供一維平均線設計的圖形介面。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from multall_turbomachinery_design.meangen import MeanLineSolver
from multall_turbomachinery_design.meangen.data_structures import (
    FlowType,
    GasProperties,
    InputType,
    MachineType,
    MeangenConfig,
    StageDesign,
)
from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterForm
from multall_turbomachinery_design.ui.widgets.result_display import (
    KeyValueDisplay,
    ResultTable,
    ResultText,
)

if TYPE_CHECKING:
    pass


class MeangenPanel(QWidget):
    """MEANGEN 面板元件。

    提供一維平均線設計的完整介面。
    """

    # 計算完成信號 (Qt 使用 camelCase)
    calculationFinished = Signal(dict)  # noqa: N815

    # 狀態變更信號
    statusChanged = Signal(str)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化 MEANGEN 面板。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._solver: MeanLineSolver | None = None

    def _setup_ui(self) -> None:
        """設置 UI 元件。"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 使用 splitter 分割左右區域
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        # 左側：參數輸入
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self._param_form = ParameterForm()
        left_layout.addWidget(self._param_form)

        # 機器參數
        machine_group = self._param_form.add_group(
            "machine",
            "機器參數",
            tooltip="定義渦輪機械的基本配置",
        )
        machine_group.add_combo(
            "machine_type",
            "機器類型:",
            ["渦輪", "壓縮機"],
            current=0,
            tooltip="選擇設計的渦輪機械類型\n渦輪: 將流體能量轉換為機械功\n壓縮機: 提升流體壓力",
        )
        machine_group.add_combo(
            "flow_type",
            "流動類型:",
            ["軸流", "混流", "徑向流"],
            current=0,
            tooltip="選擇流動方向類型\n軸流: 流體沿軸向流動\n混流: 流體同時有軸向和徑向分量\n徑向流: 流體主要沿徑向流動",
        )
        machine_group.add_int(
            "n_stages",
            "級數:",
            value=1,
            minimum=1,
            maximum=20,
            tooltip="渦輪機械的總級數 (1-20)",
        )
        machine_group.add_float(
            "rpm",
            "轉速:",
            value=10000.0,
            minimum=100.0,
            maximum=100000.0,
            suffix="rpm",
            tooltip="轉子轉速 (每分鐘轉數)",
        )
        machine_group.add_float(
            "mass_flow",
            "質量流率:",
            value=10.0,
            minimum=0.01,
            maximum=1000.0,
            suffix="kg/s",
            tooltip="工作流體的質量流率",
        )
        machine_group.add_float(
            "design_radius",
            "設計半徑:",
            value=0.3,
            minimum=0.01,
            maximum=10.0,
            suffix="m",
            tooltip="平均線設計的參考半徑 (通常為平均半徑)",
        )

        # 氣體參數
        gas_group = self._param_form.add_group(
            "gas",
            "氣體參數",
            tooltip="定義工作流體的熱力學性質",
        )
        gas_group.add_float(
            "gamma",
            "比熱比 γ:",
            value=1.4,
            minimum=1.0,
            maximum=2.0,
            tooltip="等熵指數 (cp/cv)\n空氣: 1.4\n燃氣: 1.33",
        )
        gas_group.add_float(
            "rgas",
            "氣體常數 R:",
            value=287.05,
            minimum=100.0,
            maximum=1000.0,
            suffix="J/(kg·K)",
            tooltip="理想氣體常數\n空氣: 287.05 J/(kg·K)",
        )
        gas_group.add_float(
            "poin",
            "入口總壓:",
            value=1.0,
            minimum=0.01,
            maximum=100.0,
            suffix="bar",
            tooltip="入口滯止壓力",
        )
        gas_group.add_float(
            "toin",
            "入口總溫:",
            value=1200.0,
            minimum=200.0,
            maximum=2000.0,
            suffix="K",
            tooltip="入口滯止溫度",
        )

        # 級設計參數
        stage_group = self._param_form.add_group(
            "stage",
            "級設計參數",
            tooltip="定義每級的氣動設計參數",
        )
        stage_group.add_combo(
            "input_type",
            "輸入類型:",
            ["Type A (φ, ψ, R)", "Type B (α1, α2, α3)"],
            current=0,
            tooltip="參數輸入方式\nType A: 使用無因次係數 (推薦)\nType B: 使用絕對流動角度",
        )
        stage_group.add_float(
            "phi",
            "流量係數 φ:",
            value=0.6,
            minimum=0.1,
            maximum=2.0,
            tooltip="φ = Vx / U\n軸向速度與葉尖速度之比\n典型值: 0.4-0.8",
        )
        stage_group.add_float(
            "psi",
            "負荷係數 ψ:",
            value=2.0,
            minimum=0.1,
            maximum=5.0,
            tooltip="ψ = Δh0 / U²\n焓變與葉尖速度平方之比\n渦輪典型值: 1.5-2.5\n壓縮機典型值: 0.3-0.5",
        )
        stage_group.add_float(
            "reaction",
            "反應度 R:",
            value=0.5,
            minimum=0.0,
            maximum=1.0,
            tooltip="轉子靜焓升與級靜焓升之比\n50% 反應度設計是最常見的選擇",
        )
        stage_group.add_float(
            "efficiency",
            "等熵效率:",
            value=0.90,
            minimum=0.5,
            maximum=1.0,
            tooltip="設計點的等熵效率\n典型值: 0.88-0.92",
        )

        self._param_form.add_stretch()

        # 按鈕列
        button_layout = QHBoxLayout()

        self._run_btn = QPushButton("運行計算")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setToolTip("執行平均線設計計算 (F5)")

        self._reset_btn = QPushButton("重置參數")
        self._reset_btn.setToolTip("將所有參數重置為預設值")

        self._export_btn = QPushButton("輸出檔案")
        self._export_btn.setToolTip("將計算結果輸出為 STAGEN 輸入檔案")

        button_layout.addWidget(self._run_btn)
        button_layout.addWidget(self._reset_btn)
        button_layout.addWidget(self._export_btn)
        left_layout.addLayout(button_layout)

        splitter.addWidget(left_widget)

        # 右側：結果顯示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 性能摘要
        self._perf_display = KeyValueDisplay()
        perf_group = self._create_group("性能摘要", self._perf_display)
        right_layout.addWidget(perf_group)

        # 速度三角形
        self._triangle_table = ResultTable("速度三角形", show_export=True)
        self._triangle_table.set_headers(
            ["級", "站", "Vx (m/s)", "Vt (m/s)", "V (m/s)", "α (°)", "W (m/s)", "β (°)"]
        )
        right_layout.addWidget(self._triangle_table)

        # 日誌輸出
        self._log_text = ResultText("計算日誌", show_export=True)
        right_layout.addWidget(self._log_text)

        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])

    def _create_group(self, title: str, widget: QWidget) -> QWidget:
        """建立帶標題的群組。"""
        from PySide6.QtWidgets import QGroupBox

        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _connect_signals(self) -> None:
        """連接信號與槽。"""
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)

    @Slot()
    def _on_run_clicked(self) -> None:
        """處理運行按鈕點擊。"""
        try:
            self._log_text.clear_text()
            self._log_text.append_info("開始計算...")
            self.statusChanged.emit("計算中...")

            # 建立配置
            config = self._create_config()
            self._log_text.append_text(f"機器類型: {config.machine_type.name}")
            self._log_text.append_text(f"流動類型: {config.flow_type.name}")
            self._log_text.append_text(f"級數: {config.nstages}")
            self._log_text.append_text(f"轉速: {config.rpm:.0f} rpm")
            self._log_text.append_text(f"質量流率: {config.mass_flow:.2f} kg/s")

            # 建立求解器
            self._solver = MeanLineSolver(config)

            # 運行計算
            result = self._solver.run()

            # 顯示結果
            self._display_results(result)

            self._log_text.append_success("\n計算完成！")
            self.statusChanged.emit("計算完成")
            self.calculationFinished.emit(result)

        except Exception as e:
            self._log_text.append_error(f"\n錯誤: {e}")
            self.statusChanged.emit("計算錯誤")
            QMessageBox.critical(self, "計算錯誤", str(e))

    def _create_config(self) -> MeangenConfig:
        """從 UI 建立配置物件。"""
        values = self._param_form.get_all_values()
        machine = values["machine"]
        gas_vals = values["gas"]
        stage_vals = values["stage"]

        # 機器類型
        machine_type = (
            MachineType.TURBINE if machine["machine_type"] == 0 else MachineType.COMPRESSOR
        )

        # 流動類型
        flow_types = [FlowType.AXIAL, FlowType.MIXED, FlowType.RADIAL]
        flow_type = flow_types[machine["flow_type"]]

        # 氣體性質
        gas = GasProperties(
            gamma=gas_vals["gamma"],
            rgas=gas_vals["rgas"],
            poin=gas_vals["poin"],
            toin=gas_vals["toin"],
        )

        # 配置
        config = MeangenConfig(
            machine_type=machine_type,
            flow_type=flow_type,
            gas=gas,
            nstages=machine["n_stages"],
            rpm=machine["rpm"],
            mass_flow=machine["mass_flow"],
            design_radius=machine["design_radius"],
        )

        # 級設計
        input_type = InputType.TYPE_A if stage_vals["input_type"] == 0 else InputType.TYPE_B

        for i in range(machine["n_stages"]):
            stage = StageDesign(
                stage_number=i + 1,
                input_type=input_type,
                phi=stage_vals["phi"],
                psi=stage_vals["psi"],
                reaction=stage_vals["reaction"],
                r_design=machine["design_radius"],
                efficiency=stage_vals["efficiency"],
            )
            config.stages.append(stage)

        return config

    def _display_results(self, result: dict) -> None:
        """顯示計算結果。"""
        # 性能摘要
        power = abs(result.get("power", 0))
        perf_data = {
            "功率 (kW)": power / 1000,
            "總壓比": result.get("pressure_ratio", 1.0),
            "總溫比": result.get("temperature_ratio", 1.0),
            "等熵效率": result.get("efficiency", 0.0),
            "質量流率 (kg/s)": result.get("mass_flow", 0.0),
        }
        self._perf_display.set_data(perf_data)

        # 顯示日誌
        self._log_text.append_text("")
        self._log_text.append_info("=== 性能結果 ===")
        self._log_text.append_text(f"功率: {power / 1000:.2f} kW")
        self._log_text.append_text(f"總壓比: {result.get('pressure_ratio', 1.0):.4f}")
        self._log_text.append_text(f"等熵效率: {result.get('efficiency', 0.0):.4f}")

        # 速度三角形
        self._triangle_table.clear_data()
        triangles = result.get("velocity_triangles", [])
        for tri in triangles:
            self._triangle_table.add_row(
                [
                    str(tri.get("stage", "")),
                    str(tri.get("station", "")),
                    f"{tri.get('vx', 0):.1f}",
                    f"{tri.get('vt', 0):.1f}",
                    f"{tri.get('v', 0):.1f}",
                    f"{tri.get('alpha', 0):.2f}",
                    f"{tri.get('w', 0):.1f}",
                    f"{tri.get('beta', 0):.2f}",
                ]
            )

    @Slot()
    def _on_reset_clicked(self) -> None:
        """處理重置按鈕點擊。"""
        # 重置為預設值
        machine_group = self._param_form.get_group("machine")
        if machine_group:
            machine_group.set_value("machine_type", 0)
            machine_group.set_value("flow_type", 0)
            machine_group.set_value("n_stages", 1)
            machine_group.set_value("rpm", 10000.0)
            machine_group.set_value("mass_flow", 10.0)
            machine_group.set_value("design_radius", 0.3)

        gas_group = self._param_form.get_group("gas")
        if gas_group:
            gas_group.set_value("gamma", 1.4)
            gas_group.set_value("rgas", 287.05)
            gas_group.set_value("poin", 1.0)
            gas_group.set_value("toin", 1200.0)

        stage_group = self._param_form.get_group("stage")
        if stage_group:
            stage_group.set_value("input_type", 0)
            stage_group.set_value("phi", 0.6)
            stage_group.set_value("psi", 2.0)
            stage_group.set_value("reaction", 0.5)
            stage_group.set_value("efficiency", 0.90)

        # 清除結果
        self._perf_display.clear_data()
        self._triangle_table.clear_data()
        self._log_text.clear_text()

        self.statusChanged.emit("參數已重置")

    @Slot()
    def _on_export_clicked(self) -> None:
        """處理輸出按鈕點擊。"""
        if self._solver is None:
            QMessageBox.warning(self, "警告", "請先執行計算")
            return

        # 選擇輸出目錄
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出目錄",
            "",
        )

        if not dir_path:
            return

        try:
            # 輸出檔案
            from pathlib import Path

            output_dir = Path(dir_path)
            self._solver.write_outputs(output_dir)

            self._log_text.append_success(f"\n檔案已輸出到: {output_dir}")
            self.statusChanged.emit("檔案輸出完成")
            QMessageBox.information(self, "完成", f"檔案已輸出到:\n{output_dir}")
        except Exception as e:
            self._log_text.append_error(f"\n輸出錯誤: {e}")
            QMessageBox.critical(self, "輸出錯誤", str(e))

    def get_solver(self) -> MeanLineSolver | None:
        """取得求解器實例。"""
        return self._solver

    def get_state(self) -> dict:
        """獲取面板狀態（用於專案儲存）。

        Returns:
            包含所有參數值的字典
        """
        return self._param_form.get_all_values()

    def set_state(self, state: dict) -> None:
        """設置面板狀態（用於專案載入）。

        Args:
            state: 參數值字典
        """
        for group_name, group_values in state.items():
            group = self._param_form.get_group(group_name)
            if group and isinstance(group_values, dict):
                for key, value in group_values.items():
                    try:
                        group.set_value(key, value)
                    except Exception:
                        pass  # 忽略無效的參數

    def reset(self) -> None:
        """重置面板到初始狀態。"""
        self._on_reset_clicked()
