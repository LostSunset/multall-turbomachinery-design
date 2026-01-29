# -*- coding: utf-8 -*-
"""主視窗介面。

提供 MULTALL 渦輪機械設計系統的主要圖形使用者介面。
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTabWidget,
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


class MainWindow(QMainWindow):
    """主視窗類別。"""

    def __init__(self) -> None:
        """初始化主視窗。"""
        super().__init__()
        self.setWindowTitle("MULTALL 渦輪機械設計系統")
        self.setMinimumSize(1400, 900)

        # 專案管理器
        self._project_manager = ProjectManager()

        # 建立中央 widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 建立佈局
        layout = QVBoxLayout(self.central_widget)

        # 建立標籤頁
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 新增各模組面板
        self._setup_panels()

        # 建立選單列
        self._setup_menu()

        # 建立狀態列
        self.statusBar().showMessage("就緒")

        # 建立新專案
        self._project_manager.new_project()

    def _setup_panels(self) -> None:
        """設置模組面板。"""
        # MEANGEN 面板
        self._meangen_panel = MeangenPanel()
        self._meangen_panel.statusChanged.connect(self._on_status_changed)
        self.tabs.addTab(self._meangen_panel, "MEANGEN - 平均線設計")

        # STAGEN 面板
        self._stagen_panel = StagenPanel()
        self._stagen_panel.statusChanged.connect(self._on_status_changed)
        self.tabs.addTab(self._stagen_panel, "STAGEN - 葉片幾何")

        # MULTALL 面板
        self._multall_panel = MultallPanel()
        self._multall_panel.statusChanged.connect(self._on_status_changed)
        self.tabs.addTab(self._multall_panel, "MULTALL - 3D 求解器")

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

        exit_action = QAction("結束(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 編輯選單
        edit_menu = menubar.addMenu("編輯(&E)")

        reset_action = QAction("重置參數(&R)", self)
        reset_action.triggered.connect(self._on_reset_params)
        edit_menu.addAction(reset_action)

        # 工具選單
        tools_menu = menubar.addMenu("工具(&T)")

        run_action = QAction("運行計算(&R)", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._on_run_calculation)
        tools_menu.addAction(run_action)

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

        help_menu.addSeparator()

        about_action = QAction("關於(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        """處理狀態變更。"""
        self.statusBar().showMessage(status)

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
        self.statusBar().showMessage("已新建專案")

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
                self.statusBar().showMessage(f"已開啟: {file_path}")

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
            self.statusBar().showMessage(f"已儲存: {file_path}")

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
                self.statusBar().showMessage(f"已儲存: {saved_path}")

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
        current = self.tabs.currentWidget()
        if hasattr(current, "_on_reset_clicked"):
            current._on_reset_clicked()

    @Slot()
    def _on_run_calculation(self) -> None:
        """處理運行計算。"""
        current = self.tabs.currentWidget()
        if hasattr(current, "_on_run_clicked"):
            current._on_run_clicked()
        elif hasattr(current, "_on_generate_clicked"):
            current._on_generate_clicked()

    @Slot()
    def _on_settings(self) -> None:
        """處理設定。"""
        QMessageBox.information(
            self,
            "設定",
            "設定功能開發中...",
        )

    @Slot()
    def _on_show_docs(self) -> None:
        """處理顯示文檔。"""
        import webbrowser

        webbrowser.open("https://github.com/LostSunset/multall-turbomachinery-design")

    def _show_about(self) -> None:
        """顯示關於對話框。"""
        QMessageBox.about(
            self,
            "關於 MULTALL",
            f"<h3>MULTALL 渦輪機械設計系統</h3>"
            f"<p>版本: {__version__}</p>"
            f"<p>基於 Python 3.14 和 PySide6 的現代化渦輪機械設計系統</p>"
            f"<p><b>功能模組:</b></p>"
            f"<ul>"
            f"<li>MEANGEN - 一維平均線設計</li>"
            f"<li>STAGEN - 葉片幾何生成與操作</li>"
            f"<li>MULTALL - 三維 Navier-Stokes 求解器</li>"
            f"</ul>"
            f"<p>原始系統來源: "
            f'<a href="https://sites.google.com/view/multall-turbomachinery-design/">'
            f"MULTALL Turbomachinery Design</a></p>"
            f"<p>授權: MIT</p>",
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

    # 建立並顯示主視窗
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
