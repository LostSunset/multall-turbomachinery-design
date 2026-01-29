# -*- coding: utf-8 -*-
"""結果顯示元件。

提供可重用的結果顯示元件：
- ResultTable: 表格顯示
- ResultText: 文字顯示
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
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

    用於顯示表格形式的結果數據。
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        """初始化結果表格。

        Args:
            title: 群組標題
            parent: 父元件
        """
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self._table = QTableWidget(self)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

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

    def add_row(self, row: Sequence[str]) -> None:
        """添加一行數據。

        Args:
            row: 行數據
        """
        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)
        for j, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, j, item)

    def clear_data(self) -> None:
        """清除表格數據。"""
        self._table.setRowCount(0)

    def get_row_count(self) -> int:
        """取得行數。"""
        return self._table.rowCount()

    def get_column_count(self) -> int:
        """取得列數。"""
        return self._table.columnCount()


class ResultText(QGroupBox):
    """結果文字元件。

    用於顯示文字形式的結果。
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        """初始化結果文字框。

        Args:
            title: 群組標題
            parent: 父元件
        """
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self._text = QTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setFontFamily("Consolas, Courier New, monospace")
        layout.addWidget(self._text)

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


class KeyValueDisplay(QWidget):
    """鍵值對顯示元件。

    用於顯示一組鍵值對數據。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        """初始化鍵值對顯示。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(self)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["項目", "值"])
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

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
            self._table.setItem(i, 0, key_item)

            # 格式化數值
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            value_item = QTableWidgetItem(value_str)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, value_item)

    def add_item(self, key: str, value: str | float | int) -> None:
        """添加一個鍵值對。

        Args:
            key: 鍵
            value: 值
        """
        row = self._table.rowCount()
        self._table.insertRow(row)

        key_item = QTableWidgetItem(str(key))
        key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, key_item)

        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        value_item = QTableWidgetItem(value_str)
        value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 1, value_item)

    def clear_data(self) -> None:
        """清除數據。"""
        self._table.setRowCount(0)
