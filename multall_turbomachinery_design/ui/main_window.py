# -*- coding: utf-8 -*-
"""主視窗介面。

提供 MULTALL 渦輪機械設計系統的主要圖形使用者介面。
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """主視窗類別。"""

    def __init__(self) -> None:
        """初始化主視窗。"""
        super().__init__()
        self.setWindowTitle("MULTALL 渦輪機械設計系統")
        self.setMinimumSize(1200, 800)

        # 建立中央 widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 建立佈局
        layout = QVBoxLayout(self.central_widget)

        # 建立標籤頁
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 新增各模組標籤頁
        self._setup_tabs()

        # 建立選單列
        self._setup_menu()

        # 建立狀態列
        self.statusBar().showMessage("就緒")

    def _setup_tabs(self) -> None:
        """設置標籤頁。"""
        # MEANGEN 標籤頁
        meangen_tab = QWidget()
        meangen_layout = QVBoxLayout(meangen_tab)
        meangen_label = QLabel("MEANGEN - 一維平均線設計模組")
        meangen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meangen_layout.addWidget(meangen_label)
        self.tabs.addTab(meangen_tab, "MEANGEN")

        # STAGEN 標籤頁
        stagen_tab = QWidget()
        stagen_layout = QVBoxLayout(stagen_tab)
        stagen_label = QLabel("STAGEN - 葉片幾何生成與操作模組")
        stagen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stagen_layout.addWidget(stagen_label)
        self.tabs.addTab(stagen_tab, "STAGEN")

        # MULTALL 標籤頁
        multall_tab = QWidget()
        multall_layout = QVBoxLayout(multall_tab)
        multall_label = QLabel("MULTALL - 三維 Navier-Stokes 求解器模組")
        multall_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        multall_layout.addWidget(multall_label)
        self.tabs.addTab(multall_tab, "MULTALL")

    def _setup_menu(self) -> None:
        """設置選單列。"""
        menubar = self.menuBar()

        # 檔案選單
        file_menu = menubar.addMenu("檔案(&F)")

        exit_action = QAction("結束(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 說明選單
        help_menu = menubar.addMenu("說明(&H)")

        about_action = QAction("關於(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        """顯示關於對話框。"""
        QMessageBox.about(
            self,
            "關於 MULTALL",
            "<h3>MULTALL 渦輪機械設計系統</h3>"
            "<p>版本: 0.1.0</p>"
            "<p>基於 Python 3.14 和 PySide6 的現代化渦輪機械設計系統</p>"
            "<p>原始系統來源: "
            '<a href="https://sites.google.com/view/multall-turbomachinery-design/">'
            "MULTALL Turbomachinery Design</a></p>"
            "<p>授權: MIT</p>",
        )


def main() -> int:
    """主程式進入點。

    Returns:
        程式退出碼
    """
    app = QApplication(sys.argv)

    # 設置應用程式資訊
    app.setApplicationName("MULTALL 渦輪機械設計系統")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("LostSunset")

    # 建立並顯示主視窗
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
