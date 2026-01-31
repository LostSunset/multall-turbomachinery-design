# -*- coding: utf-8 -*-
"""結果顯示元件。

提供可重用的結果顯示元件：
- ResultTable: 表格顯示
- ResultText: 文字顯示
- KeyValueDisplay: 鍵值對顯示
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ResultTable(QGroupBox):
    """結果表格元件。

    用於顯示表格形式的結果數據，支援匯出功能。
    """

    # 數據更新信號 (Qt 使用 camelCase)
    dataChanged = Signal()  # noqa: N815

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        show_export: bool = False,
    ) -> None:
        """初始化結果表格。

        Args:
            title: 群組標題
            parent: 父元件
            show_export: 是否顯示匯出按鈕
        """
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self._table = QTableWidget(self)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._table)

        # 匯出按鈕（可選）
        if show_export:
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()

            self._export_csv_btn = QPushButton("匯出 CSV")
            self._export_csv_btn.setToolTip("將表格數據匯出為 CSV 格式")
            self._export_csv_btn.clicked.connect(self._export_csv)
            btn_layout.addWidget(self._export_csv_btn)

            self._copy_btn = QPushButton("複製")
            self._copy_btn.setToolTip("複製表格數據到剪貼簿")
            self._copy_btn.clicked.connect(self._copy_to_clipboard)
            btn_layout.addWidget(self._copy_btn)

            layout.addLayout(btn_layout)

    def _show_context_menu(self, pos) -> None:
        """顯示右鍵選單。"""
        menu = QMenu(self)

        copy_action = menu.addAction("複製選中內容")
        copy_action.triggered.connect(self._copy_selection)

        copy_all_action = menu.addAction("複製全部")
        copy_all_action.triggered.connect(self._copy_to_clipboard)

        menu.addSeparator()

        export_action = menu.addAction("匯出 CSV...")
        export_action.triggered.connect(self._export_csv)

        menu.exec(self._table.mapToGlobal(pos))

    def _copy_selection(self) -> None:
        """複製選中的單元格。"""
        selection = self._table.selectedIndexes()
        if not selection:
            return

        # 獲取選中範圍
        rows = sorted(set(index.row() for index in selection))
        cols = sorted(set(index.column() for index in selection))

        text_lines = []
        for row in rows:
            row_data = []
            for col in cols:
                item = self._table.item(row, col)
                row_data.append(item.text() if item else "")
            text_lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(text_lines))

    def _copy_to_clipboard(self) -> None:
        """複製全部數據到剪貼簿。"""
        lines = []

        # 標頭
        headers = []
        for col in range(self._table.columnCount()):
            item = self._table.horizontalHeaderItem(col)
            headers.append(item.text() if item else "")
        lines.append("\t".join(headers))

        # 數據
        for row in range(self._table.rowCount()):
            row_data = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                row_data.append(item.text() if item else "")
            lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(lines))

    def _export_csv(self) -> None:
        """匯出為 CSV 檔案。"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出 CSV",
            "",
            "CSV 檔案 (*.csv);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        try:
            with Path(file_path).open("w", encoding="utf-8-sig") as f:
                # 標頭
                headers = []
                for col in range(self._table.columnCount()):
                    item = self._table.horizontalHeaderItem(col)
                    headers.append(item.text() if item else "")
                f.write(",".join(f'"{h}"' for h in headers) + "\n")

                # 數據
                for row in range(self._table.rowCount()):
                    row_data = []
                    for col in range(self._table.columnCount()):
                        item = self._table.item(row, col)
                        value = item.text() if item else ""
                        # 轉義引號
                        value = value.replace('"', '""')
                        row_data.append(f'"{value}"')
                    f.write(",".join(row_data) + "\n")

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "匯出錯誤", f"無法匯出檔案:\n{e}")

    def set_headers(self, headers: Sequence[str]) -> None:
        """設定表格標頭。

        Args:
            headers: 標頭列表
        """
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(list(headers))

        # 設定標頭調整模式
        header = self._table.horizontalHeader()
        for i in range(len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        if len(headers) > 0:
            header.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)

    def set_data(self, data: Sequence[Sequence[str]]) -> None:
        """設定表格數據。

        Args:
            data: 二維數據陣列
        """
        self._table.setRowCount(len(data))
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, j, item)

        self.dataChanged.emit()

    def add_row(self, row: Sequence[str], highlight: bool = False) -> None:
        """添加一行數據。

        Args:
            row: 行數據
            highlight: 是否高亮顯示
        """
        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)
        for j, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if highlight:
                item.setBackground(QColor(255, 255, 200))
            self._table.setItem(row_idx, j, item)

        self.dataChanged.emit()

    def clear_data(self) -> None:
        """清除表格數據。"""
        self._table.setRowCount(0)
        self.dataChanged.emit()

    def get_row_count(self) -> int:
        """取得行數。"""
        return self._table.rowCount()

    def get_column_count(self) -> int:
        """取得列數。"""
        return self._table.columnCount()

    def get_data(self) -> list[list[str]]:
        """取得所有數據。

        Returns:
            二維數據陣列
        """
        data = []
        for row in range(self._table.rowCount()):
            row_data = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                row_data.append(item.text() if item else "")
            data.append(row_data)
        return data

    def set_cell_color(self, row: int, col: int, color: QColor) -> None:
        """設定單元格背景顏色。

        Args:
            row: 行索引
            col: 列索引
            color: 顏色
        """
        item = self._table.item(row, col)
        if item:
            item.setBackground(color)


class ResultText(QGroupBox):
    """結果文字元件。

    用於顯示文字形式的結果，支援語法高亮和匯出。
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        show_export: bool = False,
    ) -> None:
        """初始化結果文字框。

        Args:
            title: 群組標題
            parent: 父元件
            show_export: 是否顯示匯出按鈕
        """
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self._text = QTextEdit(self)
        self._text.setReadOnly(True)

        # 設置等寬字體
        font = QFont()
        font.setFamilies(["Consolas", "Courier New", "monospace"])
        font.setPointSize(10)
        self._text.setFont(font)

        layout.addWidget(self._text)

        # 匯出按鈕（可選）
        if show_export:
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()

            self._export_btn = QPushButton("匯出")
            self._export_btn.setToolTip("將日誌匯出為文字檔案")
            self._export_btn.clicked.connect(self._export_text)
            btn_layout.addWidget(self._export_btn)

            self._clear_btn = QPushButton("清除")
            self._clear_btn.setToolTip("清除日誌內容")
            self._clear_btn.clicked.connect(self.clear_text)
            btn_layout.addWidget(self._clear_btn)

            layout.addLayout(btn_layout)

    def _export_text(self) -> None:
        """匯出文字到檔案。"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出日誌",
            "",
            "文字檔案 (*.txt);;日誌檔案 (*.log);;所有檔案 (*.*)",
        )

        if not file_path:
            return

        try:
            with Path(file_path).open("w", encoding="utf-8") as f:
                f.write(self._text.toPlainText())
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "匯出錯誤", f"無法匯出檔案:\n{e}")

    def set_text(self, text: str) -> None:
        """設定顯示文字。

        Args:
            text: 要顯示的文字
        """
        self._text.setPlainText(text)

    def append_text(self, text: str) -> None:
        """追加文字。

        Args:
            text: 要追加的文字
        """
        self._text.append(text)

    def append_info(self, text: str) -> None:
        """追加資訊文字（藍色）。"""
        self._text.append(f'<span style="color: #0078D4;">{text}</span>')

    def append_success(self, text: str) -> None:
        """追加成功文字（綠色）。"""
        self._text.append(f'<span style="color: #107C10;">{text}</span>')

    def append_warning(self, text: str) -> None:
        """追加警告文字（橙色）。"""
        self._text.append(f'<span style="color: #FF8C00;">{text}</span>')

    def append_error(self, text: str) -> None:
        """追加錯誤文字（紅色）。"""
        self._text.append(f'<span style="color: #D13438;">{text}</span>')

    def clear_text(self) -> None:
        """清除文字。"""
        self._text.clear()

    def get_text(self) -> str:
        """取得文字內容。"""
        return self._text.toPlainText()

    def set_html(self, html: str) -> None:
        """設定 HTML 內容。

        Args:
            html: HTML 內容
        """
        self._text.setHtml(html)

    def scroll_to_bottom(self) -> None:
        """滾動到底部。"""
        scrollbar = self._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class KeyValueDisplay(QWidget):
    """鍵值對顯示元件。

    用於顯示一組鍵值對數據，如性能摘要。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        compact: bool = False,
    ) -> None:
        """初始化鍵值對顯示。

        Args:
            parent: 父元件
            compact: 是否使用緊湊模式
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(self)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["項目", "值"])
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(not compact)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        if compact:
            self._table.setStyleSheet("""
                QTableWidget {
                    border: none;
                }
                QTableWidget::item {
                    padding: 4px;
                }
            """)

        layout.addWidget(self._table)

    def set_data(self, data: dict[str, str | float | int]) -> None:
        """設定顯示數據。

        Args:
            data: 鍵值對字典
        """
        self._table.setRowCount(len(data))
        for i, (key, value) in enumerate(data.items()):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            bold_font = QFont()
            bold_font.setBold(True)
            key_item.setFont(bold_font)
            self._table.setItem(i, 0, key_item)

            # 格式化數值
            if isinstance(value, float):
                if abs(value) >= 1e6 or (abs(value) < 1e-4 and value != 0):
                    value_str = f"{value:.4e}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            value_item = QTableWidgetItem(value_str)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, value_item)

    def add_item(
        self,
        key: str,
        value: str | float | int,
        highlight: bool = False,
    ) -> None:
        """添加一個鍵值對。

        Args:
            key: 鍵
            value: 值
            highlight: 是否高亮顯示
        """
        row = self._table.rowCount()
        self._table.insertRow(row)

        key_item = QTableWidgetItem(str(key))
        key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        bold_font = QFont()
        bold_font.setBold(True)
        key_item.setFont(bold_font)
        if highlight:
            key_item.setBackground(QColor(255, 255, 200))
        self._table.setItem(row, 0, key_item)

        if isinstance(value, float):
            if abs(value) >= 1e6 or (abs(value) < 1e-4 and value != 0):
                value_str = f"{value:.4e}"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        value_item = QTableWidgetItem(value_str)
        value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if highlight:
            value_item.setBackground(QColor(255, 255, 200))
        self._table.setItem(row, 1, value_item)

    def update_item(self, key: str, value: str | float | int) -> bool:
        """更新指定鍵的值。

        Args:
            key: 鍵
            value: 新值

        Returns:
            是否找到並更新了項目
        """
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() == key:
                if isinstance(value, float):
                    if abs(value) >= 1e6 or (abs(value) < 1e-4 and value != 0):
                        value_str = f"{value:.4e}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                value_item = self._table.item(row, 1)
                if value_item:
                    value_item.setText(value_str)
                    return True
        return False

    def clear_data(self) -> None:
        """清除數據。"""
        self._table.setRowCount(0)

    def get_data(self) -> dict[str, str]:
        """取得所有數據。

        Returns:
            鍵值對字典
        """
        data = {}
        for row in range(self._table.rowCount()):
            key_item = self._table.item(row, 0)
            value_item = self._table.item(row, 1)
            if key_item and value_item:
                data[key_item.text()] = value_item.text()
        return data
