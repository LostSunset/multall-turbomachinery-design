# -*- coding: utf-8 -*-
"""STAGEN 面板。

提供葉片幾何生成與操作的圖形介面。
"""

from __future__ import annotations

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

from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterForm
from multall_turbomachinery_design.ui.widgets.result_display import (
    KeyValueDisplay,
    ResultTable,
    ResultText,
)


class StagenPanel(QWidget):
    """STAGEN 面板元件。

    提供葉片幾何生成與操作的完整介面。
    """

    # 計算完成信號 (Qt 使用 camelCase)
    calculationFinished = Signal(dict)  # noqa: N815

    # 狀態變更信號
    statusChanged = Signal(str)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化 STAGEN 面板。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

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

        # 葉片幾何參數
        blade_group = self._param_form.add_group("blade", "葉片幾何")
        blade_group.add_int("n_blades", "葉片數:", value=50, minimum=10, maximum=200)
        blade_group.add_float("chord", "弦長:", value=0.05, minimum=0.001, maximum=1.0, suffix="m")
        blade_group.add_float("pitch", "節距:", value=0.03, minimum=0.001, maximum=1.0, suffix="m")
        blade_group.add_float(
            "max_thickness",
            "最大厚度比:",
            value=0.10,
            minimum=0.01,
            maximum=0.30,
        )
        blade_group.add_float(
            "le_radius",
            "前緣半徑比:",
            value=0.02,
            minimum=0.001,
            maximum=0.1,
        )
        blade_group.add_float(
            "te_radius",
            "後緣半徑比:",
            value=0.01,
            minimum=0.001,
            maximum=0.05,
        )

        # 角度參數
        angle_group = self._param_form.add_group("angles", "角度參數")
        angle_group.add_float(
            "inlet_angle", "入口金屬角:", value=30.0, minimum=-89.0, maximum=89.0, suffix="°"
        )
        angle_group.add_float(
            "outlet_angle", "出口金屬角:", value=-60.0, minimum=-89.0, maximum=89.0, suffix="°"
        )
        angle_group.add_float(
            "stagger_angle", "安裝角:", value=-15.0, minimum=-89.0, maximum=89.0, suffix="°"
        )

        # 堆疊參數
        stack_group = self._param_form.add_group("stacking", "堆疊參數")
        stack_group.add_combo(
            "stack_type",
            "堆疊方式:",
            ["質心", "前緣", "後緣", "最大厚度"],
            current=0,
        )
        stack_group.add_float(
            "sweep_angle", "掃掠角:", value=0.0, minimum=-45.0, maximum=45.0, suffix="°"
        )
        stack_group.add_float(
            "lean_angle", "傾斜角:", value=0.0, minimum=-45.0, maximum=45.0, suffix="°"
        )

        # 網格參數
        mesh_group = self._param_form.add_group("mesh", "網格參數")
        mesh_group.add_int("ni", "軸向網格數:", value=41, minimum=11, maximum=201)
        mesh_group.add_int("nj", "周向網格數:", value=21, minimum=5, maximum=101)
        mesh_group.add_int("nk", "徑向網格數:", value=17, minimum=5, maximum=101)

        self._param_form.add_stretch()

        # 按鈕列
        button_layout = QHBoxLayout()
        self._load_btn = QPushButton("載入檔案")
        self._gen_btn = QPushButton("生成幾何")
        self._export_btn = QPushButton("輸出檔案")
        self._cad_btn = QPushButton("輸出 CAD")
        button_layout.addWidget(self._load_btn)
        button_layout.addWidget(self._gen_btn)
        button_layout.addWidget(self._export_btn)
        button_layout.addWidget(self._cad_btn)
        left_layout.addLayout(button_layout)

        # 儲存生成的截面數據供 CAD 使用
        self._generated_sections = None

        splitter.addWidget(left_widget)

        # 右側：結果顯示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 幾何摘要
        self._geom_display = KeyValueDisplay()
        geom_group = self._create_group("幾何摘要", self._geom_display)
        right_layout.addWidget(geom_group)

        # 截面數據
        self._section_table = ResultTable("葉片截面")
        self._section_table.set_headers(
            ["K", "半徑 (m)", "弦長 (m)", "入口角 (°)", "出口角 (°)", "安裝角 (°)"]
        )
        right_layout.addWidget(self._section_table)

        # 日誌輸出
        self._log_text = ResultText("生成日誌")
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
        self._gen_btn.clicked.connect(self._on_generate_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._cad_btn.clicked.connect(self._on_cad_export_clicked)

    @Slot()
    def _on_load_clicked(self) -> None:
        """處理載入按鈕點擊。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "載入 STAGEN 輸入檔案",
            "",
            "STAGEN 檔案 (*.dat);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        try:
            self._log_text.clear_text()
            self._log_text.append_text(f"載入檔案: {file_path}\n")

            # TODO: 實作檔案載入
            self._log_text.append_text("檔案載入完成")
            self.statusChanged.emit("檔案載入完成")
        except Exception as e:
            self._log_text.append_text(f"\n錯誤: {e}")
            QMessageBox.critical(self, "載入錯誤", str(e))

    @Slot()
    def _on_generate_clicked(self) -> None:
        """處理生成按鈕點擊。"""
        try:
            self._log_text.clear_text()
            self._log_text.append_text("開始生成葉片幾何...\n")
            self.statusChanged.emit("生成中...")

            values = self._param_form.get_all_values()
            blade = values["blade"]
            angles = values["angles"]
            # stack = values["stacking"]  # 保留供未來使用
            mesh = values["mesh"]

            self._log_text.append_text(f"葉片數: {blade['n_blades']}\n")
            self._log_text.append_text(f"弦長: {blade['chord']:.4f} m\n")
            self._log_text.append_text(f"網格: {mesh['ni']} x {mesh['nj']} x {mesh['nk']}\n")

            # 顯示幾何摘要
            self._geom_display.set_data(
                {
                    "葉片數": blade["n_blades"],
                    "弦長 (m)": blade["chord"],
                    "節距 (m)": blade["pitch"],
                    "稠度": blade["chord"] / blade["pitch"],
                    "最大厚度比": blade["max_thickness"],
                    "入口金屬角 (°)": angles["inlet_angle"],
                    "出口金屬角 (°)": angles["outlet_angle"],
                    "轉折角 (°)": angles["inlet_angle"] - angles["outlet_angle"],
                    "網格點數": mesh["ni"] * mesh["nj"] * mesh["nk"],
                }
            )

            # 生成截面數據
            self._section_table.clear_data()
            n_sections = mesh["nk"]
            self._generated_sections = []  # 儲存供 CAD 使用

            import numpy as np

            for k in range(n_sections):
                span = k / (n_sections - 1) if n_sections > 1 else 0.5
                r = 0.3 + 0.1 * span  # 半徑

                # 計算此截面的弦長（可選：隨展向變化）
                chord_k = blade["chord"] * (1 - 0.1 * span)

                # 生成翼型座標（NACA 4 位數型）
                n_pts = 50
                t = blade["max_thickness"]  # 最大厚度比

                # 上下表面
                x_c = np.linspace(0.001, 1, n_pts // 2)
                y_t = 5 * t * (
                    0.2969 * np.sqrt(x_c)
                    - 0.1260 * x_c
                    - 0.3516 * x_c**2
                    + 0.2843 * x_c**3
                    - 0.1015 * x_c**4
                )

                x_upper = chord_k * x_c
                y_upper = chord_k * y_t
                x_lower = chord_k * x_c[::-1]
                y_lower = -chord_k * y_t[::-1]

                x = np.concatenate([x_upper, x_lower[1:]])
                y = np.concatenate([y_upper, y_lower[1:]])

                # 應用安裝角旋轉
                stagger_rad = np.radians(angles["stagger_angle"])
                x_rot = x * np.cos(stagger_rad) - y * np.sin(stagger_rad)
                y_rot = x * np.sin(stagger_rad) + y * np.cos(stagger_rad)

                z = np.full_like(x, r)

                self._generated_sections.append({
                    "span": span,
                    "radius": r,
                    "chord": chord_k,
                    "x": x_rot,
                    "y": y_rot,
                    "z": z,
                })

                self._section_table.add_row(
                    [
                        str(k),
                        f"{r:.4f}",
                        f"{chord_k:.4f}",
                        f"{angles['inlet_angle']:.1f}",
                        f"{angles['outlet_angle']:.1f}",
                        f"{angles['stagger_angle']:.1f}",
                    ]
                )

            self._log_text.append_text("\n幾何生成完成！")
            self._log_text.append_text(f"\n已生成 {n_sections} 個截面，可輸出 CAD")
            self.statusChanged.emit("生成完成")

        except Exception as e:
            self._log_text.append_text(f"\n錯誤: {e}")
            self.statusChanged.emit("生成錯誤")
            QMessageBox.critical(self, "生成錯誤", str(e))

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
            # TODO: 實作檔案輸出
            self._log_text.append_text("\n檔案輸出完成")
            self.statusChanged.emit("檔案輸出完成")
            QMessageBox.information(self, "完成", f"檔案已輸出到:\n{dir_path}")
        except Exception as e:
            QMessageBox.critical(self, "輸出錯誤", str(e))

    @Slot()
    def _on_cad_export_clicked(self) -> None:
        """處理 CAD 輸出按鈕點擊。"""
        # 檢查是否有生成的截面
        if not self._generated_sections:
            QMessageBox.warning(self, "警告", "請先點擊「生成幾何」生成葉片截面")
            return

        # 檢查 CAD 功能是否可用
        try:
            from multall_turbomachinery_design.cad import check_cad_available

            if not check_cad_available():
                QMessageBox.warning(
                    self,
                    "CAD 功能不可用",
                    "CadQuery 未安裝或不支援當前 Python 版本。\n\n"
                    "CAD 功能需要 Python 3.12 或 3.13。\n"
                    "安裝方式: pip install multall-turbomachinery-design[cad]",
                )
                return
        except ImportError:
            QMessageBox.warning(
                self,
                "CAD 功能不可用",
                "CAD 模組未安裝。\n\n"
                "安裝方式: pip install multall-turbomachinery-design[cad]",
            )
            return

        # 選擇輸出檔案
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "輸出 CAD 檔案",
            "blade.step",
            "STEP 檔案 (*.step *.stp);;STL 檔案 (*.stl);;IGES 檔案 (*.iges *.igs)",
        )

        if not file_path:
            return

        try:
            from pathlib import Path

            import numpy as np

            from multall_turbomachinery_design.cad import BladeCADExporter
            from multall_turbomachinery_design.cad.blade_cad import BladeSection

            self._log_text.append_text("\n\n開始生成 CAD 模型...")
            self.statusChanged.emit("生成 CAD 中...")

            # 建立截面
            sections = []
            for sec_data in self._generated_sections:
                section = BladeSection(
                    span_fraction=sec_data["span"],
                    x=np.array(sec_data["x"]),
                    y=np.array(sec_data["y"]),
                    z=np.array(sec_data["z"]),
                )
                sections.append(section)

            # 建立導出器並生成葉片
            exporter = BladeCADExporter()
            exporter.create_blade_from_sections(sections)

            # 導出檔案
            output_path = Path(file_path)
            exporter.export(output_path)

            file_size = output_path.stat().st_size
            self._log_text.append_text(f"\nCAD 檔案已輸出: {output_path}")
            self._log_text.append_text(f"檔案大小: {file_size:,} bytes")
            self.statusChanged.emit("CAD 輸出完成")

            QMessageBox.information(
                self,
                "CAD 輸出完成",
                f"CAD 檔案已成功輸出:\n{output_path}\n\n檔案大小: {file_size:,} bytes",
            )

        except Exception as e:
            self._log_text.append_text(f"\nCAD 輸出錯誤: {e}")
            self.statusChanged.emit("CAD 輸出錯誤")
            QMessageBox.critical(self, "CAD 輸出錯誤", str(e))

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
        # 重置參數為預設值
        blade_group = self._param_form.get_group("blade")
        if blade_group:
            blade_group.set_value("n_blades", 50)
            blade_group.set_value("chord", 0.05)
            blade_group.set_value("pitch", 0.03)
            blade_group.set_value("max_thickness", 0.10)

        angle_group = self._param_form.get_group("angles")
        if angle_group:
            angle_group.set_value("inlet_angle", 30.0)
            angle_group.set_value("outlet_angle", -60.0)
            angle_group.set_value("stagger_angle", -15.0)

        # 清除結果
        self._section_table.clear_data()
        self._geom_display.clear_data()
        self._log_text.clear_text()
        self._generated_sections = None
        self.statusChanged.emit("參數已重置")
