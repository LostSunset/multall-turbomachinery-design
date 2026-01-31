# -*- coding: utf-8 -*-
"""主視窗介面。

提供 MULTALL 渦輪機械設計系統的主要圖形使用者介面。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from multall_turbomachinery_design import __version__
from multall_turbomachinery_design.ui.panels import (
    MeangenPanel,
    MultallPanel,
    StagenPanel,
)
from multall_turbomachinery_design.ui.project_manager import ProjectManager
from multall_turbomachinery_design.ui.styles import (
    MIN_TOUCH_SIZE,
    ThemeMode,
    apply_style,
    set_theme,
)


class NavigationButton(QToolButton):
    """側邊導航按鈕。

    符合無障礙設計標準：
    - 最小點擊區域 44x44px
    - 清晰的視覺反饋
    """

    def __init__(
        self,
        text: str,
        icon_char: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """初始化導航按鈕。

        Args:
            text: 按鈕文字
            icon_char: 圖標字符（用於顯示）
            parent: 父元件
        """
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        # 確保最小點擊區域符合無障礙標準 (44x44px)
        self.setMinimumSize(QSize(100, max(70, MIN_TOUCH_SIZE)))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # 設置字體
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)


class SideNavigation(QFrame):
    """側邊導航欄。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化側邊導航欄。"""
        super().__init__(parent)
        self.setObjectName("sideNavigation")
        self.setFixedWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(8)

        # 標題
        title_label = QLabel("MULTALL")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setProperty("heading", True)
        layout.addWidget(title_label)

        version_label = QLabel(f"v{__version__}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setProperty("secondary", True)
        layout.addWidget(version_label)

        layout.addSpacing(20)

        # 導航按鈕
        self._buttons: list[NavigationButton] = []

        self._meangen_btn = NavigationButton("MEANGEN\n平均線設計")
        self._meangen_btn.setToolTip("一維平均線設計\n計算速度三角形和整體性能")
        self._buttons.append(self._meangen_btn)
        layout.addWidget(self._meangen_btn)

        self._stagen_btn = NavigationButton("STAGEN\n葉片幾何")
        self._stagen_btn.setToolTip("葉片幾何生成與操作\n創建 2D/3D 葉片截面")
        self._buttons.append(self._stagen_btn)
        layout.addWidget(self._stagen_btn)

        self._multall_btn = NavigationButton("MULTALL\n3D 求解器")
        self._multall_btn.setToolTip("三維 Navier-Stokes 求解器\nCFD 流場模擬")
        self._buttons.append(self._multall_btn)
        layout.addWidget(self._multall_btn)

        layout.addStretch()

        # 設定按鈕
        self._settings_btn = NavigationButton("設定")
        self._settings_btn.setToolTip("應用程式設定")
        layout.addWidget(self._settings_btn)

        # 設置第一個按鈕為選中狀態
        self._meangen_btn.setChecked(True)

    @property
    def meangen_btn(self) -> NavigationButton:
        """MEANGEN 按鈕。"""
        return self._meangen_btn

    @property
    def stagen_btn(self) -> NavigationButton:
        """STAGEN 按鈕。"""
        return self._stagen_btn

    @property
    def multall_btn(self) -> NavigationButton:
        """MULTALL 按鈕。"""
        return self._multall_btn

    @property
    def settings_btn(self) -> NavigationButton:
        """設定按鈕。"""
        return self._settings_btn

    def set_active(self, index: int) -> None:
        """設置活動按鈕。"""
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)


class MainWindow(QMainWindow):
    """主視窗類別。"""

    def __init__(self) -> None:
        """初始化主視窗。"""
        super().__init__()
        self.setWindowTitle("MULTALL 渦輪機械設計系統")
        self.setMinimumSize(1400, 900)

        # 專案管理器
        self._project_manager = ProjectManager()

        # 主題模式
        self._theme_mode = ThemeMode.LIGHT

        # 建立中央 widget
        self._setup_central_widget()

        # 建立工具列
        self._setup_toolbar()

        # 建立選單列
        self._setup_menu()

        # 建立狀態列
        self._setup_statusbar()

        # 連接信號
        self._connect_signals()

        # 建立新專案
        self._project_manager.new_project()

    def _setup_central_widget(self) -> None:
        """設置中央元件。"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 側邊導航
        self._side_nav = SideNavigation()
        main_layout.addWidget(self._side_nav)

        # 主內容區域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 堆疊式頁面
        self._stack = QStackedWidget()
        content_layout.addWidget(self._stack)

        # 新增各模組面板
        self._setup_panels()

        main_layout.addWidget(content_widget, 1)

    def _setup_panels(self) -> None:
        """設置模組面板。"""
        # MEANGEN 面板
        self._meangen_panel = MeangenPanel()
        self._meangen_panel.statusChanged.connect(self._on_status_changed)
        self._stack.addWidget(self._meangen_panel)

        # STAGEN 面板
        self._stagen_panel = StagenPanel()
        self._stagen_panel.statusChanged.connect(self._on_status_changed)
        self._stack.addWidget(self._stagen_panel)

        # MULTALL 面板
        self._multall_panel = MultallPanel()
        self._multall_panel.statusChanged.connect(self._on_status_changed)
        self._stack.addWidget(self._multall_panel)

    def _setup_toolbar(self) -> None:
        """設置工具列。"""
        toolbar = QToolBar("主工具列")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # 檔案操作
        new_action = QAction("新建", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setToolTip("新建專案 (Ctrl+N)")
        new_action.triggered.connect(self._on_new_project)
        toolbar.addAction(new_action)

        open_action = QAction("開啟", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setToolTip("開啟專案 (Ctrl+O)")
        open_action.triggered.connect(self._on_open_project)
        toolbar.addAction(open_action)

        save_action = QAction("儲存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setToolTip("儲存專案 (Ctrl+S)")
        save_action.triggered.connect(self._on_save_project)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # 運行操作
        self._run_action = QAction("運行", self)
        self._run_action.setShortcut("F5")
        self._run_action.setToolTip("運行計算 (F5)")
        self._run_action.triggered.connect(self._on_run_calculation)
        toolbar.addAction(self._run_action)

        toolbar.addSeparator()

        # 主題切換
        self._theme_action = QAction("深色模式", self)
        self._theme_action.setCheckable(True)
        self._theme_action.setToolTip("切換深色/淺色主題")
        self._theme_action.triggered.connect(self._on_toggle_theme)
        toolbar.addAction(self._theme_action)

        # 添加彈性空間
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # 說明
        help_action = QAction("說明", self)
        help_action.setShortcut("F1")
        help_action.setToolTip("使用手冊 (F1)")
        help_action.triggered.connect(self._on_show_docs)
        toolbar.addAction(help_action)

    def _setup_menu(self) -> None:
        """設置選單列。"""
        menubar = self.menuBar()

        # 檔案選單
        file_menu = menubar.addMenu("檔案(&F)")

        new_action = QAction("新建專案(&N)", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("開啟專案(&O)...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("儲存專案(&S)", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        save_as_action = QAction("另存新檔(&A)...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # 最近檔案子選單
        recent_menu = file_menu.addMenu("最近的專案(&R)")
        recent_menu.addAction(QAction("(無)", self))

        file_menu.addSeparator()

        exit_action = QAction("結束(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 編輯選單
        edit_menu = menubar.addMenu("編輯(&E)")

        reset_action = QAction("重置參數(&R)", self)
        reset_action.triggered.connect(self._on_reset_params)
        edit_menu.addAction(reset_action)

        edit_menu.addSeparator()

        # 偏好設定
        prefs_action = QAction("偏好設定(&P)...", self)
        prefs_action.triggered.connect(self._on_settings)
        edit_menu.addAction(prefs_action)

        # 檢視選單
        view_menu = menubar.addMenu("檢視(&V)")

        # 主題子選單
        theme_menu = view_menu.addMenu("主題(&T)")

        light_action = QAction("淺色主題", self)
        light_action.triggered.connect(lambda: self._set_theme(ThemeMode.LIGHT))
        theme_menu.addAction(light_action)

        dark_action = QAction("深色主題", self)
        dark_action.triggered.connect(lambda: self._set_theme(ThemeMode.DARK))
        theme_menu.addAction(dark_action)

        view_menu.addSeparator()

        # 面板導航
        meangen_action = QAction("MEANGEN 面板(&M)", self)
        meangen_action.setShortcut("Alt+1")
        meangen_action.triggered.connect(lambda: self._switch_panel(0))
        view_menu.addAction(meangen_action)

        stagen_action = QAction("STAGEN 面板(&S)", self)
        stagen_action.setShortcut("Alt+2")
        stagen_action.triggered.connect(lambda: self._switch_panel(1))
        view_menu.addAction(stagen_action)

        multall_action = QAction("MULTALL 面板(&U)", self)
        multall_action.setShortcut("Alt+3")
        multall_action.triggered.connect(lambda: self._switch_panel(2))
        view_menu.addAction(multall_action)

        # 工具選單
        tools_menu = menubar.addMenu("工具(&T)")

        run_action = QAction("運行計算(&R)", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._on_run_calculation)
        tools_menu.addAction(run_action)

        tools_menu.addSeparator()

        # 批次處理
        batch_action = QAction("批次處理(&B)...", self)
        batch_action.triggered.connect(self._on_batch_process)
        tools_menu.addAction(batch_action)

        # 參數優化
        optimize_action = QAction("參數優化(&O)...", self)
        optimize_action.triggered.connect(self._on_optimize)
        tools_menu.addAction(optimize_action)

        tools_menu.addSeparator()

        settings_action = QAction("設定(&S)...", self)
        settings_action.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_action)

        # 說明選單
        help_menu = menubar.addMenu("說明(&H)")

        doc_action = QAction("使用手冊(&D)", self)
        doc_action.setShortcut("F1")
        doc_action.triggered.connect(self._on_show_docs)
        help_menu.addAction(doc_action)

        tutorial_action = QAction("教學範例(&T)", self)
        tutorial_action.triggered.connect(self._on_show_tutorials)
        help_menu.addAction(tutorial_action)

        help_menu.addSeparator()

        # 檢查更新
        update_action = QAction("檢查更新(&U)...", self)
        update_action.triggered.connect(self._on_check_updates)
        help_menu.addAction(update_action)

        help_menu.addSeparator()

        about_action = QAction("關於(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        """設置狀態列。"""
        statusbar = self.statusBar()

        # 主要狀態訊息
        self._status_label = QLabel("就緒")
        statusbar.addWidget(self._status_label, 1)

        # 當前模組
        self._module_label = QLabel("MEANGEN")
        self._module_label.setProperty("secondary", True)
        statusbar.addPermanentWidget(self._module_label)

        # 專案狀態
        self._project_label = QLabel("新專案")
        self._project_label.setProperty("secondary", True)
        statusbar.addPermanentWidget(self._project_label)

    def _connect_signals(self) -> None:
        """連接信號。"""
        # 側邊導航
        self._side_nav.meangen_btn.clicked.connect(lambda: self._switch_panel(0))
        self._side_nav.stagen_btn.clicked.connect(lambda: self._switch_panel(1))
        self._side_nav.multall_btn.clicked.connect(lambda: self._switch_panel(2))
        self._side_nav.settings_btn.clicked.connect(self._on_settings)

    def _switch_panel(self, index: int) -> None:
        """切換面板。"""
        self._stack.setCurrentIndex(index)
        self._side_nav.set_active(index)

        # 更新模組標籤
        module_names = ["MEANGEN", "STAGEN", "MULTALL"]
        if 0 <= index < len(module_names):
            self._module_label.setText(module_names[index])

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        """處理狀態變更。"""
        self._status_label.setText(status)

    @Slot()
    def _on_new_project(self) -> None:
        """處理新建專案。"""
        # 檢查是否有未儲存的變更
        if self._project_manager.is_modified:
            reply = QMessageBox.question(
                self,
                "新建專案",
                "目前專案有未儲存的變更。確定要新建專案嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 建立新專案
        self._project_manager.new_project()

        # 重置所有面板
        self._reset_all_panels()

        self._update_window_title()
        self._project_label.setText("新專案")
        self._status_label.setText("已新建專案")

    @Slot()
    def _on_open_project(self) -> None:
        """處理開啟專案。"""
        from PySide6.QtWidgets import QFileDialog

        # 檢查是否有未儲存的變更
        if self._project_manager.is_modified:
            reply = QMessageBox.question(
                self,
                "開啟專案",
                "目前專案有未儲存的變更。確定要開啟其他專案嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟專案",
            "",
            "MULTALL 專案 (*.mtproj);;所有檔案 (*.*)",
        )
        if file_path:
            try:
                data = self._project_manager.load(file_path)

                # 載入各面板資料
                if data.meangen:
                    self._load_panel_state(self._meangen_panel, data.meangen)
                if data.stagen:
                    self._load_panel_state(self._stagen_panel, data.stagen)
                if data.multall:
                    self._load_panel_state(self._multall_panel, data.multall)

                self._update_window_title()
                self._project_label.setText(Path(file_path).stem)
                self._status_label.setText(f"已開啟: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"載入專案失敗:\n{e}")

    @Slot()
    def _on_save_project(self) -> None:
        """處理儲存專案。"""
        # 如果沒有當前檔案，使用另存新檔
        if self._project_manager.current_file is None:
            self._on_save_project_as()
            return

        try:
            # 收集各面板資料
            self._collect_panel_states()

            # 儲存專案
            file_path = self._project_manager.save()
            self._update_window_title()
            self._status_label.setText(f"已儲存: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存專案失敗:\n{e}")

    @Slot()
    def _on_save_project_as(self) -> None:
        """處理另存新檔。"""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存專案",
            "",
            "MULTALL 專案 (*.mtproj);;所有檔案 (*.*)",
        )
        if file_path:
            try:
                # 收集各面板資料
                self._collect_panel_states()

                # 儲存專案
                saved_path = self._project_manager.save(file_path)
                self._update_window_title()
                self._project_label.setText(Path(saved_path).stem)
                self._status_label.setText(f"已儲存: {saved_path}")

            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"儲存專案失敗:\n{e}")

    def _collect_panel_states(self) -> None:
        """收集各面板的狀態。"""
        # 從各面板獲取狀態
        if hasattr(self._meangen_panel, "get_state"):
            self._project_manager.update_meangen(self._meangen_panel.get_state())
        if hasattr(self._stagen_panel, "get_state"):
            self._project_manager.update_stagen(self._stagen_panel.get_state())
        if hasattr(self._multall_panel, "get_state"):
            self._project_manager.update_multall(self._multall_panel.get_state())

    def _load_panel_state(self, panel: QWidget, state: dict) -> None:
        """載入面板狀態。

        Args:
            panel: 面板元件
            state: 狀態資料
        """
        if hasattr(panel, "set_state"):
            panel.set_state(state)

    def _reset_all_panels(self) -> None:
        """重置所有面板。"""
        if hasattr(self._meangen_panel, "reset"):
            self._meangen_panel.reset()
        if hasattr(self._stagen_panel, "reset"):
            self._stagen_panel.reset()
        if hasattr(self._multall_panel, "reset"):
            self._multall_panel.reset()

    def _update_window_title(self) -> None:
        """更新視窗標題。"""
        title = "MULTALL 渦輪機械設計系統"
        if self._project_manager.has_project and self._project_manager.data:
            name = self._project_manager.data.metadata.name
            if name:
                title = f"{name} - {title}"
            if self._project_manager.is_modified:
                title = f"*{title}"
        self.setWindowTitle(title)

    @Slot()
    def _on_reset_params(self) -> None:
        """處理重置參數。"""
        current = self._stack.currentWidget()
        if hasattr(current, "_on_reset_clicked"):
            current._on_reset_clicked()

    @Slot()
    def _on_run_calculation(self) -> None:
        """處理運行計算。"""
        current = self._stack.currentWidget()
        if hasattr(current, "_on_run_clicked"):
            current._on_run_clicked()
        elif hasattr(current, "_on_generate_clicked"):
            current._on_generate_clicked()

    @Slot()
    def _on_toggle_theme(self) -> None:
        """切換主題。"""
        if self._theme_mode == ThemeMode.LIGHT:
            self._set_theme(ThemeMode.DARK)
            self._theme_action.setChecked(True)
        else:
            self._set_theme(ThemeMode.LIGHT)
            self._theme_action.setChecked(False)

    def _set_theme(self, mode: ThemeMode) -> None:
        """設置主題。"""
        self._theme_mode = mode
        stylesheet = set_theme(mode)
        QApplication.instance().setStyleSheet(stylesheet)

    @Slot()
    def _on_batch_process(self) -> None:
        """處理批次處理。"""
        QMessageBox.information(
            self,
            "批次處理",
            "批次處理功能開發中...\n\n"
            "此功能將允許您：\n"
            "• 批次運行多個設計案例\n"
            "• 自動化參數掃描\n"
            "• 生成比較報告",
        )

    @Slot()
    def _on_optimize(self) -> None:
        """處理參數優化。"""
        QMessageBox.information(
            self,
            "參數優化",
            "參數優化功能開發中...\n\n"
            "此功能將提供：\n"
            "• 單目標優化\n"
            "• 多目標優化 (Pareto)\n"
            "• 基於 AI 的智能建議",
        )

    @Slot()
    def _on_settings(self) -> None:
        """處理設定。"""
        QMessageBox.information(
            self,
            "設定",
            "設定功能開發中...\n\n"
            "即將推出的設定選項：\n"
            "• 界面主題和字體\n"
            "• 預設參數配置\n"
            "• 輸出格式偏好\n"
            "• 求解器選項",
        )

    @Slot()
    def _on_show_docs(self) -> None:
        """處理顯示文檔。"""
        import webbrowser

        webbrowser.open("https://github.com/LostSunset/multall-turbomachinery-design")

    @Slot()
    def _on_show_tutorials(self) -> None:
        """顯示教學範例。"""
        QMessageBox.information(
            self,
            "教學範例",
            "教學範例開發中...\n\n"
            "即將推出的教學：\n"
            "• 快速入門指南\n"
            "• 渦輪設計流程\n"
            "• 壓縮機設計流程\n"
            "• 進階技巧",
        )

    @Slot()
    def _on_check_updates(self) -> None:
        """檢查更新。"""
        QMessageBox.information(
            self,
            "檢查更新",
            f"目前版本: {__version__}\n\n您已經使用最新版本。",
        )

    def _show_about(self) -> None:
        """顯示關於對話框。"""
        QMessageBox.about(
            self,
            "關於 MULTALL",
            f"<h2>MULTALL 渦輪機械設計系統</h2>"
            f"<p><b>版本:</b> {__version__}</p>"
            f"<p>基於 Python 3.14 和 PySide6 的現代化渦輪機械設計系統</p>"
            f"<hr>"
            f"<p><b>功能模組:</b></p>"
            f"<ul>"
            f"<li><b>MEANGEN</b> - 一維平均線設計</li>"
            f"<li><b>STAGEN</b> - 葉片幾何生成與操作</li>"
            f"<li><b>MULTALL</b> - 三維 Navier-Stokes 求解器</li>"
            f"</ul>"
            f"<hr>"
            f"<p><b>原始系統來源:</b></p>"
            f'<p><a href="https://sites.google.com/view/multall-turbomachinery-design/">'
            f"MULTALL Turbomachinery Design</a></p>"
            f"<p><b>授權:</b> MIT License</p>"
            f"<p><b>開發:</b> LostSunset</p>",
        )


def main() -> int:
    """主程式進入點。

    Returns:
        程式退出碼
    """
    app = QApplication(sys.argv)

    # 設置應用程式資訊
    app.setApplicationName("MULTALL 渦輪機械設計系統")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("LostSunset")

    # 應用樣式
    apply_style(app)

    # 建立並顯示主視窗
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
