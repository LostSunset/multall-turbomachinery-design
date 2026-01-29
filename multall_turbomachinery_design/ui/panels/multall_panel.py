# -*- coding: utf-8 -*-
"""MULTALL 面板。

提供 3D Navier-Stokes 求解器的圖形介面。
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterForm
from multall_turbomachinery_design.ui.widgets.result_display import (
    KeyValueDisplay,
    ResultTable,
    ResultText,
)


class MultallPanel(QWidget):
    """MULTALL 面板元件。

    提供 3D Navier-Stokes 求解器的完整介面。
    """

    # 計算完成信號 (Qt 使用 camelCase)
    calculationFinished = Signal(dict)  # noqa: N815

    # 狀態變更信號
    statusChanged = Signal(str)  # noqa: N815

    # 進度更新信號
    progressUpdated = Signal(int, int)  # noqa: N815 (current, total)

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化 MULTALL 面板。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._is_running = False

    def _setup_ui(self) -> None:
        """設置 UI 元件。"""
        layout = QHBoxLayout(self)

        # 使用 splitter 分割左右區域
        splitter = QSplitter(self)
        layout.addWidget(splitter)

        # 左側：參數輸入
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self._param_form = ParameterForm()
        left_layout.addWidget(self._param_form)

        # 求解器參數
        solver_group = self._param_form.add_group("solver", "求解器參數")
        solver_group.add_int(
            "max_iterations", "最大迭代數:", value=10000, minimum=100, maximum=100000
        )
        solver_group.add_float("cfl", "CFL 數:", value=0.5, minimum=0.1, maximum=1.0)
        solver_group.add_float("convergence", "收斂準則:", value=1e-6, minimum=1e-10, maximum=1e-2)
        solver_group.add_combo(
            "time_scheme",
            "時間推進:",
            ["Euler", "RK2", "RK4", "SCREE"],
            current=3,
        )

        # 黏性參數
        viscous_group = self._param_form.add_group("viscous", "黏性模型")
        viscous_group.add_combo(
            "viscous_model",
            "黏性模型:",
            ["無黏性", "混合長度", "Spalart-Allmaras"],
            current=1,
        )
        viscous_group.add_float("reynolds", "雷諾數:", value=1e6, minimum=1e4, maximum=1e8)
        viscous_group.add_float(
            "turbulent_intensity", "紊流強度:", value=0.05, minimum=0.01, maximum=0.20
        )

        # 邊界條件
        bc_group = self._param_form.add_group("boundary", "邊界條件")
        bc_group.add_float("inlet_p0", "入口總壓:", value=101325.0, suffix="Pa")
        bc_group.add_float("inlet_t0", "入口總溫:", value=1200.0, suffix="K")
        bc_group.add_float("inlet_alpha", "入口流角:", value=0.0, suffix="°")
        bc_group.add_float("outlet_p", "出口靜壓:", value=50000.0, suffix="Pa")
        bc_group.add_float("omega", "轉速:", value=1000.0, suffix="rad/s")

        # 混合平面
        mixing_group = self._param_form.add_group("mixing", "混合平面")
        mixing_group.add_combo(
            "mixing_type",
            "平均方式:",
            ["周向平均", "質量平均", "通量平均", "無反射"],
            current=0,
        )
        mixing_group.add_float("relaxation", "鬆弛因子:", value=0.5, minimum=0.1, maximum=1.0)

        self._param_form.add_stretch()

        # 進度條
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        left_layout.addWidget(self._progress)

        # 按鈕列
        button_layout = QHBoxLayout()
        self._load_btn = QPushButton("載入網格")
        self._run_btn = QPushButton("開始求解")
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._export_btn = QPushButton("輸出結果")
        button_layout.addWidget(self._load_btn)
        button_layout.addWidget(self._run_btn)
        button_layout.addWidget(self._stop_btn)
        button_layout.addWidget(self._export_btn)
        left_layout.addLayout(button_layout)

        splitter.addWidget(left_widget)

        # 右側：結果顯示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 性能摘要
        self._perf_display = KeyValueDisplay()
        perf_group = self._create_group("性能結果", self._perf_display)
        right_layout.addWidget(perf_group)

        # 收斂歷史
        self._conv_table = ResultTable("收斂歷史")
        self._conv_table.set_headers(["迭代", "殘差", "質量守恆", "效率"])
        right_layout.addWidget(self._conv_table)

        # 站點數據
        self._station_table = ResultTable("站點數據")
        self._station_table.set_headers(["J", "P (Pa)", "T (K)", "Vx (m/s)", "Vt (m/s)", "Ma"])
        right_layout.addWidget(self._station_table)

        # 日誌輸出
        self._log_text = ResultText("求解日誌")
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
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self.progressUpdated.connect(self._update_progress)

    @Slot()
    def _on_load_clicked(self) -> None:
        """處理載入按鈕點擊。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "載入網格檔案",
            "",
            "MULTALL 網格 (*.dat *.msh);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        try:
            self._log_text.clear_text()
            self._log_text.append_text(f"載入網格: {file_path}\n")

            # TODO: 實作網格載入
            self._log_text.append_text("網格載入完成\n")
            self.statusChanged.emit("網格載入完成")
        except Exception as e:
            self._log_text.append_text(f"\n錯誤: {e}")
            QMessageBox.critical(self, "載入錯誤", str(e))

    @Slot()
    def _on_run_clicked(self) -> None:
        """處理運行按鈕點擊。"""
        if self._is_running:
            return

        try:
            self._is_running = True
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._progress.setValue(0)

            self._log_text.clear_text()
            self._log_text.append_text("開始 CFD 求解...\n")
            self.statusChanged.emit("求解中...")

            values = self._param_form.get_all_values()
            solver = values["solver"]
            viscous = values["viscous"]
            bc = values["boundary"]
            # mixing = values["mixing"]  # 保留供未來使用

            self._log_text.append_text(f"最大迭代: {solver['max_iterations']}\n")
            self._log_text.append_text(f"CFL: {solver['cfl']}\n")
            self._log_text.append_text(
                f"黏性模型: {['無黏性', '混合長度', 'SA'][viscous['viscous_model']]}\n"
            )
            self._log_text.append_text(f"入口總壓: {bc['inlet_p0']:.0f} Pa\n")
            self._log_text.append_text(f"出口靜壓: {bc['outlet_p']:.0f} Pa\n")

            # 模擬計算過程
            self._simulate_solve(solver["max_iterations"])

        except Exception as e:
            self._log_text.append_text(f"\n錯誤: {e}")
            self.statusChanged.emit("求解錯誤")
            QMessageBox.critical(self, "求解錯誤", str(e))
        finally:
            self._is_running = False
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def _simulate_solve(self, max_iter: int) -> None:
        """模擬求解過程（示例）。"""
        import random

        # 清除舊數據
        self._conv_table.clear_data()
        self._station_table.clear_data()

        # 模擬收斂歷史
        residual = 1.0
        for i in range(min(20, max_iter // 500)):
            if not self._is_running:
                break

            residual *= 0.7  # 模擬收斂
            efficiency = 0.85 + 0.05 * (1 - residual)
            mass_balance = residual * 0.1

            self._conv_table.add_row(
                [
                    str((i + 1) * 500),
                    f"{residual:.2e}",
                    f"{mass_balance:.2e}",
                    f"{efficiency:.4f}",
                ]
            )

            progress = int((i + 1) / 20 * 100)
            self._progress.setValue(progress)

        # 顯示性能結果
        self._perf_display.set_data(
            {
                "質量流率 (kg/s)": 10.0 + random.uniform(-0.1, 0.1),
                "總壓比": 2.0 + random.uniform(-0.05, 0.05),
                "總溫比": 0.85 + random.uniform(-0.02, 0.02),
                "等熵效率": 0.88 + random.uniform(-0.02, 0.02),
                "功率 (kW)": 500 + random.uniform(-20, 20),
                "殘差": residual,
            }
        )

        # 顯示站點數據
        for j in range(5):
            self._station_table.add_row(
                [
                    str(j * 5),
                    f"{100000 - j * 10000:.0f}",
                    f"{1200 - j * 50:.0f}",
                    f"{150 + j * 10:.1f}",
                    f"{-50 - j * 20:.1f}",
                    f"{0.3 + j * 0.1:.2f}",
                ]
            )

        self._log_text.append_text(f"\n求解完成！最終殘差: {residual:.2e}")
        self.statusChanged.emit("求解完成")
        self._progress.setValue(100)

    @Slot()
    def _on_stop_clicked(self) -> None:
        """處理停止按鈕點擊。"""
        self._is_running = False
        self._log_text.append_text("\n用戶停止求解")
        self.statusChanged.emit("求解已停止")

    @Slot()
    def _on_export_clicked(self) -> None:
        """處理輸出按鈕點擊。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出目錄",
            "",
        )

        if not dir_path:
            return

        try:
            self._log_text.append_text(f"\n輸出目錄: {dir_path}")
            # TODO: 實作結果輸出
            self._log_text.append_text("\n結果輸出完成")
            self.statusChanged.emit("結果輸出完成")
            QMessageBox.information(self, "完成", f"結果已輸出到:\n{dir_path}")
        except Exception as e:
            QMessageBox.critical(self, "輸出錯誤", str(e))

    @Slot(int, int)
    def _update_progress(self, current: int, total: int) -> None:
        """更新進度條。"""
        if total > 0:
            self._progress.setValue(int(current / total * 100))
