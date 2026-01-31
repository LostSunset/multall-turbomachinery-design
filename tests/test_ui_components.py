# -*- coding: utf-8 -*-
"""UI 元件單元測試。

測試各個 UI 元件的獨立功能。
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

    _app = QApplication.instance() or QApplication([])
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False

pytestmark = pytest.mark.skipif(
    not HAS_PYTEST_QT or not HAS_DISPLAY,
    reason="需要 pytest-qt 和顯示器環境",
)


class TestFloatSpinBoxComponent:
    """FloatSpinBox 元件單元測試。"""

    def test_initialization_default(self, qtbot) -> None:
        """測試預設初始化。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox()
        qtbot.addWidget(spin)

        assert spin.value() == 0.0
        assert spin.decimals() == 4
        assert spin.minimum() == -1e10
        assert spin.maximum() == 1e10

    def test_initialization_custom(self, qtbot) -> None:
        """測試自定義初始化。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(
            minimum=-100.0,
            maximum=100.0,
            decimals=2,
            step=0.5,
            value=25.0,
            suffix="mm",
            tooltip="測試提示",
        )
        qtbot.addWidget(spin)

        assert spin.value() == 25.0
        assert spin.minimum() == -100.0
        assert spin.maximum() == 100.0
        assert spin.decimals() == 2
        assert spin.singleStep() == 0.5
        assert "mm" in spin.suffix()
        assert spin.toolTip() == "測試提示"

    def test_value_change(self, qtbot) -> None:
        """測試值變更。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=0.0, maximum=100.0, value=50.0)
        qtbot.addWidget(spin)

        spin.setValue(75.0)
        assert spin.value() == 75.0

    def test_range_enforcement(self, qtbot) -> None:
        """測試範圍限制。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=0.0, maximum=100.0, value=50.0)
        qtbot.addWidget(spin)

        spin.setValue(150.0)  # 超過最大值
        assert spin.value() == 100.0

        spin.setValue(-50.0)  # 低於最小值
        assert spin.value() == 0.0

    def test_range_tooltip_auto_generated(self, qtbot) -> None:
        """測試自動生成的範圍提示。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=0.0, maximum=100.0)
        qtbot.addWidget(spin)

        # 沒有提供 tooltip 時應自動生成範圍提示
        assert "範圍" in spin.toolTip()
        assert "0" in spin.toolTip()
        assert "100" in spin.toolTip()


class TestIntSpinBoxComponent:
    """IntSpinBox 元件單元測試。"""

    def test_initialization_default(self, qtbot) -> None:
        """測試預設初始化。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox()
        qtbot.addWidget(spin)

        assert spin.value() == 0
        assert spin.minimum() == 0

    def test_initialization_custom(self, qtbot) -> None:
        """測試自定義初始化。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(
            minimum=1,
            maximum=100,
            step=5,
            value=50,
            suffix="個",
            tooltip="數量提示",
        )
        qtbot.addWidget(spin)

        assert spin.value() == 50
        assert spin.minimum() == 1
        assert spin.maximum() == 100
        assert spin.singleStep() == 5
        assert spin.toolTip() == "數量提示"

    def test_value_change(self, qtbot) -> None:
        """測試值變更。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(minimum=0, maximum=100, value=50)
        qtbot.addWidget(spin)

        spin.setValue(75)
        assert spin.value() == 75


class TestParameterGroupComponent:
    """ParameterGroup 元件單元測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試群組", tooltip="群組提示")
        qtbot.addWidget(group)

        assert group.title() == "測試群組"
        assert group.toolTip() == "群組提示"

    def test_add_float_parameter(self, qtbot) -> None:
        """測試添加浮點數參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float(
            "pressure",
            "壓力:",
            tooltip="入口壓力",
            value=101325.0,
            suffix="Pa",
        )

        assert spin.value() == 101325.0
        assert group.get_value("pressure") == 101325.0
        assert "壓力" in str(type(spin).__name__) or spin.value() == 101325.0

    def test_add_int_parameter(self, qtbot) -> None:
        """測試添加整數參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_int("blades", "葉片數:", value=12)

        assert spin.value() == 12
        assert group.get_value("blades") == 12

    def test_add_combo_parameter(self, qtbot) -> None:
        """測試添加下拉選擇參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        combo = group.add_combo(
            "material",
            "材料:",
            ["鋁合金", "鈦合金", "不銹鋼"],
            current=1,
        )

        assert combo.currentIndex() == 1
        assert combo.currentText() == "鈦合金"
        assert group.get_value("material") == 1

    def test_add_label_parameter(self, qtbot) -> None:
        """測試添加標籤參數。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        label = group.add_label("status", "狀態:", "就緒")

        assert label.text() == "就緒"
        assert group.get_value("status") == "就緒"

    def test_set_and_get_value(self, qtbot) -> None:
        """測試設定和取得值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        group.add_float("temp", "溫度:", value=300.0)
        assert group.get_value("temp") == 300.0

        group.set_value("temp", 350.0)
        assert group.get_value("temp") == 350.0

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

    def test_set_enabled(self, qtbot) -> None:
        """測試設定啟用狀態。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float("test", "測試:", value=1.0)
        assert spin.isEnabled()

        group.set_enabled("test", False)
        assert not spin.isEnabled()

        group.set_enabled("test", True)
        assert spin.isEnabled()

    def test_validation(self, qtbot) -> None:
        """測試驗證功能。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        group.add_float("a", "A:", minimum=0.0, maximum=100.0, value=50.0)
        group.add_int("b", "B:", minimum=1, maximum=10, value=5)

        valid, errors = group.validate()
        assert valid
        assert len(errors) == 0


class TestParameterFormComponent:
    """ParameterForm 元件單元測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

    def test_add_groups(self, qtbot) -> None:
        """測試添加多個群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group1 = form.add_group("basic", "基本參數")
        group2 = form.add_group("advanced", "進階參數")

        assert group1.title() == "基本參數"
        assert group2.title() == "進階參數"
        assert form.get_group("basic") is group1
        assert form.get_group("advanced") is group2

    def test_get_all_values(self, qtbot) -> None:
        """測試獲取所有群組的值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group1 = form.add_group("inlet", "入口條件")
        group1.add_float("pressure", "壓力:", value=101325.0)
        group1.add_float("temperature", "溫度:", value=288.15)

        group2 = form.add_group("outlet", "出口條件")
        group2.add_float("back_pressure", "背壓:", value=50000.0)

        values = form.get_all_values()
        assert values["inlet"]["pressure"] == 101325.0
        assert values["inlet"]["temperature"] == 288.15
        assert values["outlet"]["back_pressure"] == 50000.0

    def test_validate_all(self, qtbot) -> None:
        """測試驗證所有群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group = form.add_group("test", "測試")
        group.add_float("value", "值:", minimum=0.0, maximum=100.0, value=50.0)

        valid, errors = form.validate_all()
        assert valid
        assert len(errors) == 0


class TestResultTableComponent:
    """ResultTable 元件單元測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("性能結果")
        qtbot.addWidget(table)

        assert table.title() == "性能結果"

    def test_set_headers(self, qtbot) -> None:
        """測試設定標頭。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["參數", "數值", "單位"])
        assert table.get_column_count() == 3

    def test_set_data(self, qtbot) -> None:
        """測試設定數據。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["名稱", "值"])
        table.set_data(
            [
                ["效率", "0.85"],
                ["功率", "1000 kW"],
            ]
        )
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
        assert table.get_row_count() == 1

        table.clear_data()
        assert table.get_row_count() == 0


class TestResultTextComponent:
    """ResultText 元件單元測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("計算日誌")
        qtbot.addWidget(text)

        assert text.title() == "計算日誌"

    def test_set_text(self, qtbot) -> None:
        """測試設定文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("計算開始...")
        assert text.get_text() == "計算開始..."

    def test_append_text(self, qtbot) -> None:
        """測試追加文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("第一行")
        text.append_text("第二行")

        content = text.get_text()
        assert "第一行" in content
        assert "第二行" in content

    def test_clear_text(self, qtbot) -> None:
        """測試清除文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("測試")
        qtbot.addWidget(text)

        text.set_text("測試內容")
        assert text.get_text() != ""

        text.clear_text()
        assert text.get_text() == ""


class TestParameterRowComponent:
    """ParameterRow 元件單元測試。"""

    def test_creation(self, qtbot) -> None:
        """測試創建。"""
        from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterRow

        row = ParameterRow()
        qtbot.addWidget(row)

    def test_add_float(self, qtbot) -> None:
        """測試添加浮點數。"""
        from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterRow

        row = ParameterRow()
        qtbot.addWidget(row)

        spin = row.add_float("x", "X:", value=1.0)
        assert spin.value() == 1.0
        assert row.get_value("x") == 1.0

    def test_add_int(self, qtbot) -> None:
        """測試添加整數。"""
        from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterRow

        row = ParameterRow()
        qtbot.addWidget(row)

        spin = row.add_int("n", "N:", value=10)
        assert spin.value() == 10
        assert row.get_value("n") == 10

    def test_add_combo(self, qtbot) -> None:
        """測試添加下拉選擇。"""
        from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterRow

        row = ParameterRow()
        qtbot.addWidget(row)

        combo = row.add_combo("type", "類型:", ["A", "B", "C"], current=1)
        assert combo.currentIndex() == 1
        assert row.get_value("type") == 1

    def test_get_all_values(self, qtbot) -> None:
        """測試獲取所有值。"""
        from multall_turbomachinery_design.ui.widgets.parameter_input import ParameterRow

        row = ParameterRow()
        qtbot.addWidget(row)

        row.add_float("x", "X:", value=1.0)
        row.add_int("n", "N:", value=5)

        values = row.get_all_values()
        assert values["x"] == 1.0
        assert values["n"] == 5
