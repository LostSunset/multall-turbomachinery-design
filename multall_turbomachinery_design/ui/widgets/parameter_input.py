# -*- coding: utf-8 -*-
"""參數輸入元件。

提供可重用的參數輸入元件：
- FloatSpinBox: 浮點數輸入
- IntSpinBox: 整數輸入
- ParameterForm: 參數表單
- ParameterGroup: 參數群組
- ParameterRow: 橫向參數列
"""

from __future__ import annotations

from typing import Any, override

from PySide6.QtCore import Qt, Signal
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

    提供更好的預設值和範圍設定，支援工具提示和驗證。
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
        tooltip: str = "",
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
            tooltip: 工具提示文字
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setValue(value)
        if suffix:
            self.setSuffix(f" {suffix}")

        # 儲存原始範圍用於驗證（必須在 _update_range_tooltip 之前設置）
        self._min = minimum
        self._max = maximum

        # 設置工具提示
        if tooltip:
            self.setToolTip(tooltip)
        else:
            # 自動生成範圍提示
            self._update_range_tooltip()

    def _update_range_tooltip(self) -> None:
        """更新範圍提示。"""
        min_str = f"{self._min:.4g}" if abs(self._min) < 1e6 else f"{self._min:.2e}"
        max_str = f"{self._max:.4g}" if abs(self._max) < 1e6 else f"{self._max:.2e}"
        tip = f"範圍: {min_str} ~ {max_str}"
        self.setToolTip(tip)

    @override
    def setRange(self, minimum: float, maximum: float) -> None:
        """設置範圍並更新提示。"""
        super().setRange(minimum, maximum)
        self._min = minimum
        self._max = maximum
        self._update_range_tooltip()


class IntSpinBox(QSpinBox):
    """增強型整數輸入框。

    提供更好的預設值和範圍設定，支援工具提示和驗證。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        minimum: int = 0,
        maximum: int = 1000000,
        step: int = 1,
        value: int = 0,
        suffix: str = "",
        tooltip: str = "",
    ) -> None:
        """初始化整數輸入框。

        Args:
            parent: 父元件
            minimum: 最小值
            maximum: 最大值
            step: 步進值
            value: 初始值
            suffix: 後綴字串
            tooltip: 工具提示文字
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setSingleStep(step)
        self.setValue(value)
        if suffix:
            self.setSuffix(f" {suffix}")

        # 設置工具提示
        if tooltip:
            self.setToolTip(tooltip)
        else:
            self.setToolTip(f"範圍: {minimum} ~ {maximum}")


class ParameterGroup(QGroupBox):
    """參數群組元件。

    將相關參數組織在一個群組框中，支援工具提示和標籤。
    """

    # 當參數變更時發出信號 (Qt 使用 camelCase)
    parameterChanged = Signal(str, object)  # noqa: N815

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        tooltip: str = "",
    ) -> None:
        """初始化參數群組。

        Args:
            title: 群組標題
            parent: 父元件
            tooltip: 群組工具提示
        """
        super().__init__(title, parent)
        self._layout = QFormLayout(self)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        # 設置適當的間距避免元件擠在一起
        self._layout.setVerticalSpacing(8)
        self._layout.setHorizontalSpacing(12)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._widgets: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}

        if tooltip:
            self.setToolTip(tooltip)

    def add_float(
        self,
        name: str,
        label: str,
        tooltip: str = "",
        **kwargs: Any,
    ) -> FloatSpinBox:
        """添加浮點數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            tooltip: 工具提示
            **kwargs: FloatSpinBox 參數

        Returns:
            建立的輸入框
        """
        # 建立標籤
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
            kwargs["tooltip"] = tooltip

        spin = FloatSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addRow(lbl, spin)
        self._widgets[name] = spin
        self._labels[name] = lbl
        return spin

    def add_int(
        self,
        name: str,
        label: str,
        tooltip: str = "",
        **kwargs: Any,
    ) -> IntSpinBox:
        """添加整數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            tooltip: 工具提示
            **kwargs: IntSpinBox 參數

        Returns:
            建立的輸入框
        """
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
            kwargs["tooltip"] = tooltip

        spin = IntSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addRow(lbl, spin)
        self._widgets[name] = spin
        self._labels[name] = lbl
        return spin

    def add_combo(
        self,
        name: str,
        label: str,
        items: list[str],
        current: int = 0,
        tooltip: str = "",
    ) -> QComboBox:
        """添加下拉選擇參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            items: 選項列表
            current: 初始選擇索引
            tooltip: 工具提示

        Returns:
            建立的下拉框
        """
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)

        combo = QComboBox(self)
        combo.addItems(items)
        combo.setCurrentIndex(current)
        combo.currentIndexChanged.connect(lambda i: self.parameterChanged.emit(name, i))

        if tooltip:
            combo.setToolTip(tooltip)

        self._layout.addRow(lbl, combo)
        self._widgets[name] = combo
        self._labels[name] = lbl
        return combo

    def add_label(
        self,
        name: str,
        label: str,
        value: str = "",
        tooltip: str = "",
    ) -> QLabel:
        """添加唯讀標籤。

        Args:
            name: 參數名稱
            label: 顯示標籤
            value: 初始值
            tooltip: 工具提示

        Returns:
            建立的標籤
        """
        lbl = QLabel(label)
        value_lbl = QLabel(value, self)

        if tooltip:
            lbl.setToolTip(tooltip)
            value_lbl.setToolTip(tooltip)

        self._layout.addRow(lbl, value_lbl)
        self._widgets[name] = value_lbl
        self._labels[name] = lbl
        return value_lbl

    def add_separator(self) -> None:
        """添加分隔線。"""
        from PySide6.QtWidgets import QFrame

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self._layout.addRow(line)

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

    def set_enabled(self, name: str, enabled: bool) -> None:
        """設定參數是否啟用。

        Args:
            name: 參數名稱
            enabled: 是否啟用
        """
        widget = self._widgets.get(name)
        if widget:
            widget.setEnabled(enabled)

        label = self._labels.get(name)
        if label:
            label.setEnabled(enabled)

    def get_all_values(self) -> dict[str, Any]:
        """取得所有參數值。

        Returns:
            參數名稱到值的映射
        """
        return {name: self.get_value(name) for name in self._widgets}

    def validate(self) -> tuple[bool, list[str]]:
        """驗證所有參數。

        Returns:
            (是否有效, 錯誤訊息列表)
        """
        errors = []
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                value = widget.value()
                if value < widget.minimum() or value > widget.maximum():
                    errors.append(f"{name}: 值超出範圍")
            elif isinstance(widget, QSpinBox):
                value = widget.value()
                if value < widget.minimum() or value > widget.maximum():
                    errors.append(f"{name}: 值超出範圍")

        return len(errors) == 0, errors


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
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(12)
        self._groups: dict[str, ParameterGroup] = {}

    def add_group(
        self,
        name: str,
        title: str,
        tooltip: str = "",
    ) -> ParameterGroup:
        """添加參數群組。

        Args:
            name: 群組名稱
            title: 群組標題
            tooltip: 群組工具提示

        Returns:
            建立的參數群組
        """
        group = ParameterGroup(title, self, tooltip)
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

    def validate_all(self) -> tuple[bool, dict[str, list[str]]]:
        """驗證所有群組的參數。

        Returns:
            (是否全部有效, 群組名稱到錯誤列表的映射)
        """
        all_valid = True
        all_errors = {}

        for name, group in self._groups.items():
            valid, errors = group.validate()
            if not valid:
                all_valid = False
                all_errors[name] = errors

        return all_valid, all_errors


class ParameterRow(QWidget):
    """橫向參數列元件。

    用於在一行中顯示多個參數。
    """

    # 當參數變更時發出信號 (Qt 使用 camelCase)
    parameterChanged = Signal(str, object)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        """初始化參數列。

        Args:
            parent: 父元件
        """
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._widgets: dict[str, QWidget] = {}

    def add_float(
        self,
        name: str,
        label: str,
        tooltip: str = "",
        **kwargs: Any,
    ) -> FloatSpinBox:
        """添加浮點數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            tooltip: 工具提示
            **kwargs: FloatSpinBox 參數

        Returns:
            建立的輸入框
        """
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
            kwargs["tooltip"] = tooltip

        self._layout.addWidget(lbl)
        spin = FloatSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addWidget(spin)
        self._widgets[name] = spin
        return spin

    def add_int(
        self,
        name: str,
        label: str,
        tooltip: str = "",
        **kwargs: Any,
    ) -> IntSpinBox:
        """添加整數參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            tooltip: 工具提示
            **kwargs: IntSpinBox 參數

        Returns:
            建立的輸入框
        """
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
            kwargs["tooltip"] = tooltip

        self._layout.addWidget(lbl)
        spin = IntSpinBox(self, **kwargs)
        spin.valueChanged.connect(lambda v: self.parameterChanged.emit(name, v))
        self._layout.addWidget(spin)
        self._widgets[name] = spin
        return spin

    def add_combo(
        self,
        name: str,
        label: str,
        items: list[str],
        current: int = 0,
        tooltip: str = "",
    ) -> QComboBox:
        """添加下拉選擇參數。

        Args:
            name: 參數名稱
            label: 顯示標籤
            items: 選項列表
            current: 初始選擇索引
            tooltip: 工具提示

        Returns:
            建立的下拉框
        """
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)

        self._layout.addWidget(lbl)
        combo = QComboBox(self)
        combo.addItems(items)
        combo.setCurrentIndex(current)
        combo.currentIndexChanged.connect(lambda i: self.parameterChanged.emit(name, i))

        if tooltip:
            combo.setToolTip(tooltip)

        self._layout.addWidget(combo)
        self._widgets[name] = combo
        return combo

    def add_stretch(self) -> None:
        """添加彈性空間。"""
        self._layout.addStretch()

    def get_value(self, name: str) -> Any:
        """取得參數值。"""
        widget = self._widgets.get(name)
        if widget is None:
            return None
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentIndex()
        return None

    def set_value(self, name: str, value: Any) -> None:
        """設定參數值。"""
        widget = self._widgets.get(name)
        if widget is None:
            return
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            widget.setValue(value)
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(value)

    def get_all_values(self) -> dict[str, Any]:
        """取得所有參數值。"""
        return {name: self.get_value(name) for name in self._widgets}
