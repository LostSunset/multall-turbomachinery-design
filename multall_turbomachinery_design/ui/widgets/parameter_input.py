# -*- coding: utf-8 -*-
"""參數輸入元件。

提供可重用的參數輸入元件：
- FloatSpinBox: 浮點數輸入
- IntSpinBox: 整數輸入
- ParameterForm: 參數表單
- ParameterGroup: 參數群組
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FloatSpinBox(QDoubleSpinBox):
    """增強型浮點數輸入框。

    提供更好的預設值和範圍設定。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        minimum: float = -1e10,
        maximum: float = 1e10,
        decimals: int = 4,
        step: float = 0.1,
        value: float = 0.0,
        suffix: str = "",
    ) -> None:
        """初始化浮點數輸入框。

        Args:
            parent: 父元件
            minimum: 最小值
            maximum: 最大值
            decimals: 小數位數
            step: 步進值
            value: 初始值
            suffix: 後綴字串（如單位）
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setValue(value)
        if suffix:
            self.setSuffix(f" {suffix}")


class IntSpinBox(QSpinBox):
    """增強型整數輸入框。

    提供更好的預設值和範圍設定。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        minimum: int = 0,
        maximum: int = 1000000,
        step: int = 1,
        value: int = 0,
        suffix: str = "",
    ) -> None:
        """初始化整數輸入框。

        Args:
            parent: 父元件
            minimum: 最小值
            maximum: 最大值
            step: 步進值
            value: 初始值
            suffix: 後綴字串
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setSingleStep(step)
        self.setValue(value)
        if suffix:
            self.setSuffix(f" {suffix}")


class ParameterGroup(QGroupBox):
    """參數群組元件。

    將相關參數組織在一個群組框中。
    """

    # 當參數變更時發出信號 (Qt 使用 camelCase)
    parameterChanged = Signal(str, object)  # noqa: N815

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        """初始化參數群組。

        Args:
            title: 群組標題
            parent: 父元件
        """
        super().__init__(title, parent)
        self._layout = QFormLayout(self)
        self._widgets: dict[str, QWidget] = {}

    def add_float(
        self,
        name: str,
        label: str,
        **kwargs: Any,
    ) -> FloatSpinBox:
        """添加浮點數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            **kwargs: FloatSpinBox 參數

        Returns:
            建立的輸入框
        """
        spin = FloatSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addRow(label, spin)
        self._widgets[name] = spin
        return spin

    def add_int(
        self,
        name: str,
        label: str,
        **kwargs: Any,
    ) -> IntSpinBox:
        """添加整數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            **kwargs: IntSpinBox 參數

        Returns:
            建立的輸入框
        """
        spin = IntSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addRow(label, spin)
        self._widgets[name] = spin
        return spin

    def add_combo(
        self,
        name: str,
        label: str,
        items: list[str],
        current: int = 0,
    ) -> QComboBox:
        """添加下拉選擇參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            items: 選項列表
            current: 初始選擇索引

        Returns:
            建立的下拉框
        """
        combo = QComboBox(self)
        combo.addItems(items)
        combo.setCurrentIndex(current)
        combo.currentIndexChanged.connect(lambda i: self.parameterChanged.emit(name, i))
        self._layout.addRow(label, combo)
        self._widgets[name] = combo
        return combo

    def add_label(
        self,
        name: str,
        label: str,
        value: str = "",
    ) -> QLabel:
        """添加唯讀標籤。

        Args:
            name: 參數名稱
            label: 顯示標籤
            value: 初始值

        Returns:
            建立的標籤
        """
        lbl = QLabel(value, self)
        self._layout.addRow(label, lbl)
        self._widgets[name] = lbl
        return lbl

    def get_value(self, name: str) -> Any:
        """取得參數值。

        Args:
            name: 參數名稱

        Returns:
            參數值
        """
        widget = self._widgets.get(name)
        if widget is None:
            return None
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentIndex()
        if isinstance(widget, QLabel):
            return widget.text()
        return None

    def set_value(self, name: str, value: Any) -> None:
        """設定參數值。

        Args:
            name: 參數名稱
            value: 新的值
        """
        widget = self._widgets.get(name)
        if widget is None:
            return
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            widget.setValue(value)
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(value)
        elif isinstance(widget, QLabel):
            widget.setText(str(value))

    def get_all_values(self) -> dict[str, Any]:
        """取得所有參數值。

        Returns:
            參數名稱到值的映射
        """
        return {name: self.get_value(name) for name in self._widgets}


class ParameterForm(QWidget):
    """參數表單元件。

    組織多個參數群組的表單。
    """

    # 當任何參數變更時發出信號 (Qt 使用 camelCase)
    parameterChanged = Signal(str, str, object)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化參數表單。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._groups: dict[str, ParameterGroup] = {}

    def add_group(self, name: str, title: str) -> ParameterGroup:
        """添加參數群組。

        Args:
            name: 群組名稱
            title: 群組標題

        Returns:
            建立的參數群組
        """
        group = ParameterGroup(title, self)
        group.parameterChanged.connect(
            lambda param, value: self.parameterChanged.emit(name, param, value)
        )
        self._layout.addWidget(group)
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> ParameterGroup | None:
        """取得參數群組。

        Args:
            name: 群組名稱

        Returns:
            參數群組，若不存在則為 None
        """
        return self._groups.get(name)

    def get_all_values(self) -> dict[str, dict[str, Any]]:
        """取得所有參數值。

        Returns:
            群組名稱到參數值的映射
        """
        return {name: group.get_all_values() for name, group in self._groups.items()}

    def add_stretch(self) -> None:
        """添加彈性空間。"""
        self._layout.addStretch()


class ParameterRow(QWidget):
    """橫向參數列元件。

    用於在一行中顯示多個參數。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化參數列。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._widgets: dict[str, QWidget] = {}

    def add_float(
        self,
        name: str,
        label: str,
        **kwargs: Any,
    ) -> FloatSpinBox:
        """添加浮點數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            **kwargs: FloatSpinBox 參數

        Returns:
            建立的輸入框
        """
        self._layout.addWidget(QLabel(label))
        spin = FloatSpinBox(self, **kwargs)
        self._layout.addWidget(spin)
        self._widgets[name] = spin
        return spin

    def add_int(
        self,
        name: str,
        label: str,
        **kwargs: Any,
    ) -> IntSpinBox:
        """添加整數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            **kwargs: IntSpinBox 參數

        Returns:
            建立的輸入框
        """
        self._layout.addWidget(QLabel(label))
        spin = IntSpinBox(self, **kwargs)
        self._layout.addWidget(spin)
        self._widgets[name] = spin
        return spin

    def add_stretch(self) -> None:
        """添加彈性空間。"""
        self._layout.addStretch()

    def get_value(self, name: str) -> Any:
        """取得參數值。"""
        widget = self._widgets.get(name)
        if widget and isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        return None
