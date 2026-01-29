# -*- coding: utf-8 -*-
"""UI 元件測試。

此測試需要 pytest-qt 和顯示器環境。
在沒有 pytest-qt 或無頭環境下會自動跳過。
"""

from __future__ import annotations

import pytest

# 檢查是否有 pytest-qt
try:
    import pytestqt  # noqa: F401

    HAS_PYTEST_QT = True
except ImportError:
    HAS_PYTEST_QT = False

# 檢查是否可以創建 Qt 應用
try:
    from PySide6.QtWidgets import QApplication

    # 嘗試創建應用程式（需要顯示器）
    _app = QApplication.instance() or QApplication([])
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False

# 如果沒有 pytest-qt 或顯示器，跳過所有測試
pytestmark = pytest.mark.skipif(
    not HAS_PYTEST_QT or not HAS_DISPLAY,
    reason="需要 pytest-qt 和顯示器環境",
)


class TestFloatSpinBox:
    """FloatSpinBox 測試。"""

    def test_default_values(self, qtbot) -> None:
        """測試預設值。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox()
        qtbot.addWidget(spin)

        assert spin.value() == 0.0
        assert spin.decimals() == 4

    def test_custom_values(self, qtbot) -> None:
        """測試自定義值。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(
            minimum=0.0,
            maximum=100.0,
            decimals=2,
            step=0.5,
            value=50.0,
            suffix="m",
        )
        qtbot.addWidget(spin)

        assert spin.value() == 50.0
        assert spin.minimum() == 0.0
        assert spin.maximum() == 100.0
        assert spin.decimals() == 2
        assert spin.singleStep() == 0.5
        assert "m" in spin.suffix()


class TestIntSpinBox:
    """IntSpinBox 測試。"""

    def test_default_values(self, qtbot) -> None:
        """測試預設值。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox()
        qtbot.addWidget(spin)

        assert spin.value() == 0
        assert spin.minimum() == 0

    def test_custom_values(self, qtbot) -> None:
        """測試自定義值。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(
            minimum=1,
            maximum=100,
            step=5,
            value=50,
            suffix="個",
        )
        qtbot.addWidget(spin)

        assert spin.value() == 50
        assert spin.minimum() == 1
        assert spin.maximum() == 100
        assert spin.singleStep() == 5


class TestParameterGroup:
    """ParameterGroup 測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試群組")
        qtbot.addWidget(group)

        assert group.title() == "測試群組"

    def test_add_float(self, qtbot) -> None:
        """測試添加浮點數參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float("test_float", "測試浮點數:", value=1.5)
        assert spin.value() == 1.5
        assert group.get_value("test_float") == 1.5

    def test_add_int(self, qtbot) -> None:
        """測試添加整數參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_int("test_int", "測試整數:", value=10)
        assert spin.value() == 10
        assert group.get_value("test_int") == 10

    def test_add_combo(self, qtbot) -> None:
        """測試添加下拉選擇。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        combo = group.add_combo("test_combo", "測試選擇:", ["A", "B", "C"], current=1)
        assert combo.currentIndex() == 1
        assert group.get_value("test_combo") == 1

    def test_add_label(self, qtbot) -> None:
        """測試添加標籤。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        label = group.add_label("test_label", "測試標籤:", "測試值")
        assert label.text() == "測試值"
        assert group.get_value("test_label") == "測試值"

    def test_set_value(self, qtbot) -> None:
        """測試設定值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        group.add_float("test", "測試:", value=1.0)
        group.set_value("test", 2.0)
        assert group.get_value("test") == 2.0

    def test_get_all_values(self, qtbot) -> None:
        """測試獲取所有值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        group.add_float("a", "A:", value=1.0)
        group.add_int("b", "B:", value=2)
        group.add_combo("c", "C:", ["X", "Y"], current=0)

        values = group.get_all_values()
        assert values["a"] == 1.0
        assert values["b"] == 2
        assert values["c"] == 0


class TestParameterForm:
    """ParameterForm 測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

    def test_add_group(self, qtbot) -> None:
        """測試添加群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group = form.add_group("test_group", "測試群組")
        assert group.title() == "測試群組"
        assert form.get_group("test_group") is group

    def test_get_all_values(self, qtbot) -> None:
        """測試獲取所有值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group1 = form.add_group("group1", "群組1")
        group1.add_float("a", "A:", value=1.0)

        group2 = form.add_group("group2", "群組2")
        group2.add_int("b", "B:", value=2)

        values = form.get_all_values()
        assert values["group1"]["a"] == 1.0
        assert values["group2"]["b"] == 2


class TestResultTable:
    """ResultTable 測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試表格")
        qtbot.addWidget(table)

        assert table.title() == "測試表格"

    def test_set_headers(self, qtbot) -> None:
        """測試設定標頭。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["A", "B", "C"])
        assert table.get_column_count() == 3

    def test_set_data(self, qtbot) -> None:
        """測試設定數據。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["A", "B"])
        table.set_data([["1", "2"], ["3", "4"]])
        assert table.get_row_count() == 2

    def test_add_row(self, qtbot) -> None:
        """測試添加行。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["A", "B"])
        table.add_row(["1", "2"])
        table.add_row(["3", "4"])
        assert table.get_row_count() == 2

    def test_clear_data(self, qtbot) -> None:
        """測試清除數據。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["A", "B"])
        table.add_row(["1", "2"])
        table.clear_data()
        assert table.get_row_count() == 0


class TestResultText:
    """ResultText 測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試文字")
        qtbot.addWidget(text)

        assert text.title() == "測試文字"

    def test_set_text(self, qtbot) -> None:
        """測試設定文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("Hello World")
        assert text.get_text() == "Hello World"

    def test_append_text(self, qtbot) -> None:
        """測試追加文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("Hello")
        text.append_text("World")
        assert "Hello" in text.get_text()
        assert "World" in text.get_text()

    def test_clear_text(self, qtbot) -> None:
        """測試清除文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("Test")
        text.clear_text()
        assert text.get_text() == ""
