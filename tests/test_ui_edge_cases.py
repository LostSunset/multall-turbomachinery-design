# -*- coding: utf-8 -*-
"""UI 邊界條件測試。

測試異常情況、邊界值和錯誤處理。
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


class TestFloatSpinBoxEdgeCases:
    """FloatSpinBox 邊界測試。"""

    def test_extreme_values(self, qtbot) -> None:
        """測試極端值。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=-1e10, maximum=1e10)
        qtbot.addWidget(spin)

        # 測試極大值
        spin.setValue(1e9)
        assert spin.value() == 1e9

        # 測試極小值
        spin.setValue(-1e9)
        assert spin.value() == -1e9

        # 測試接近零
        spin.setValue(1e-10)
        assert abs(spin.value() - 1e-10) < 1e-15

    def test_boundary_values(self, qtbot) -> None:
        """測試邊界值。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=0.0, maximum=100.0)
        qtbot.addWidget(spin)

        # 最小邊界
        spin.setValue(0.0)
        assert spin.value() == 0.0

        # 最大邊界
        spin.setValue(100.0)
        assert spin.value() == 100.0

        # 超過最大值應被限制
        spin.setValue(100.1)
        assert spin.value() == 100.0

        # 低於最小值應被限制
        spin.setValue(-0.1)
        assert spin.value() == 0.0

    def test_precision_handling(self, qtbot) -> None:
        """測試精度處理。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(decimals=6)
        qtbot.addWidget(spin)

        spin.setValue(0.123456)
        # 驗證精度
        assert abs(spin.value() - 0.123456) < 1e-7

    def test_zero_range(self, qtbot) -> None:
        """測試零範圍（最小值等於最大值）。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=50.0, maximum=50.0, value=50.0)
        qtbot.addWidget(spin)

        assert spin.value() == 50.0

        # 嘗試設置不同值應被限制
        spin.setValue(60.0)
        assert spin.value() == 50.0

    def test_negative_range(self, qtbot) -> None:
        """測試負數範圍。"""
        from multall_turbomachinery_design.ui.widgets import FloatSpinBox

        spin = FloatSpinBox(minimum=-100.0, maximum=-10.0, value=-50.0)
        qtbot.addWidget(spin)

        assert spin.value() == -50.0
        assert spin.minimum() == -100.0
        assert spin.maximum() == -10.0


class TestIntSpinBoxEdgeCases:
    """IntSpinBox 邊界測試。"""

    def test_extreme_values(self, qtbot) -> None:
        """測試極端值。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(minimum=0, maximum=1000000)
        qtbot.addWidget(spin)

        spin.setValue(999999)
        assert spin.value() == 999999

        spin.setValue(0)
        assert spin.value() == 0

    def test_boundary_values(self, qtbot) -> None:
        """測試邊界值。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(minimum=1, maximum=100)
        qtbot.addWidget(spin)

        # 最小邊界
        spin.setValue(1)
        assert spin.value() == 1

        # 最大邊界
        spin.setValue(100)
        assert spin.value() == 100

        # 超過範圍
        spin.setValue(101)
        assert spin.value() == 100

        spin.setValue(0)
        assert spin.value() == 1

    def test_step_values(self, qtbot) -> None:
        """測試步進值。"""
        from multall_turbomachinery_design.ui.widgets import IntSpinBox

        spin = IntSpinBox(minimum=0, maximum=100, step=5, value=50)
        qtbot.addWidget(spin)

        assert spin.singleStep() == 5


class TestParameterGroupEdgeCases:
    """ParameterGroup 邊界測試。"""

    def test_empty_group(self, qtbot) -> None:
        """測試空群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("空群組")
        qtbot.addWidget(group)

        values = group.get_all_values()
        assert values == {}

        valid, errors = group.validate()
        assert valid
        assert len(errors) == 0

    def test_get_nonexistent_value(self, qtbot) -> None:
        """測試取得不存在的值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        value = group.get_value("nonexistent")
        assert value is None

    def test_set_nonexistent_value(self, qtbot) -> None:
        """測試設定不存在的值。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 不應引發異常
        group.set_value("nonexistent", 100)

    def test_set_enabled_nonexistent(self, qtbot) -> None:
        """測試設定不存在參數的啟用狀態。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 不應引發異常
        group.set_enabled("nonexistent", False)

    def test_duplicate_parameter_names(self, qtbot) -> None:
        """測試重複的參數名稱。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 添加同名參數（後者會覆蓋前者）
        group.add_float("value", "值1:", value=1.0)
        group.add_float("value", "值2:", value=2.0)

        # 應該使用最後一個值
        assert group.get_value("value") == 2.0

    def test_special_characters_in_name(self, qtbot) -> None:
        """測試參數名稱中的特殊字符。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 中文名稱
        group.add_float("壓力", "壓力:", value=100.0)
        assert group.get_value("壓力") == 100.0

        # 帶空格的名稱
        group.add_float("total pressure", "總壓:", value=200.0)
        assert group.get_value("total pressure") == 200.0

        # 帶下劃線的名稱
        group.add_float("inlet_pressure", "入口壓力:", value=300.0)
        assert group.get_value("inlet_pressure") == 300.0


class TestParameterFormEdgeCases:
    """ParameterForm 邊界測試。"""

    def test_empty_form(self, qtbot) -> None:
        """測試空表單。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        values = form.get_all_values()
        assert values == {}

        valid, errors = form.validate_all()
        assert valid
        assert len(errors) == 0

    def test_get_nonexistent_group(self, qtbot) -> None:
        """測試取得不存在的群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group = form.get_group("nonexistent")
        assert group is None

    def test_many_groups(self, qtbot) -> None:
        """測試大量群組。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        # 添加 20 個群組
        for i in range(20):
            group = form.add_group(f"group_{i}", f"群組 {i}")
            group.add_float(f"param_{i}", f"參數 {i}:", value=float(i))

        values = form.get_all_values()
        assert len(values) == 20

        for i in range(20):
            assert values[f"group_{i}"][f"param_{i}"] == float(i)


class TestResultTableEdgeCases:
    """ResultTable 邊界測試。"""

    def test_empty_table(self, qtbot) -> None:
        """測試空表格。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("空表格")
        qtbot.addWidget(table)

        assert table.get_row_count() == 0

    def test_empty_data(self, qtbot) -> None:
        """測試設置空數據。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["A", "B", "C"])
        table.set_data([])

        assert table.get_row_count() == 0
        assert table.get_column_count() == 3

    def test_single_row(self, qtbot) -> None:
        """測試單行數據。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["名稱", "值"])
        table.set_data([["效率", "0.85"]])

        assert table.get_row_count() == 1

    def test_single_column(self, qtbot) -> None:
        """測試單列表格。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        table.set_headers(["參數"])
        table.set_data([["A"], ["B"], ["C"]])

        assert table.get_column_count() == 1
        assert table.get_row_count() == 3

    def test_long_text(self, qtbot) -> None:
        """測試長文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        long_text = "這是一段非常長的文字" * 10
        table.set_headers(["長標題" * 5, "值"])
        table.set_data([[long_text, "測試"]])

        assert table.get_row_count() == 1

    def test_unicode_content(self, qtbot) -> None:
        """測試 Unicode 內容。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("Unicode 測試")
        qtbot.addWidget(table)

        table.set_headers(["中文", "日文", "希臘文"])
        table.set_data([["壓力", "圧力", "πίεση"]])

        assert table.get_row_count() == 1


class TestResultTextEdgeCases:
    """ResultText 邊界測試。"""

    def test_empty_text(self, qtbot) -> None:
        """測試空文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("空文字")
        qtbot.addWidget(text)

        assert text.get_text() == ""

    def test_very_long_text(self, qtbot) -> None:
        """測試超長文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("長文字測試")
        qtbot.addWidget(text)

        long_content = "測試行\n" * 1000
        text.set_text(long_content)

        assert "測試行" in text.get_text()

    def test_special_characters(self, qtbot) -> None:
        """測試特殊字符。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("特殊字符")
        qtbot.addWidget(text)

        special = "Tab:\t換行:\n反斜線:\\ 引號:\"'"
        text.set_text(special)

        content = text.get_text()
        assert "\t" in content or "Tab" in content

    def test_unicode_text(self, qtbot) -> None:
        """測試 Unicode 文字。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("Unicode")
        qtbot.addWidget(text)

        unicode_content = "中文 日本語 한국어 العربية"
        text.set_text(unicode_content)

        assert "中文" in text.get_text()
        assert "日本語" in text.get_text()

    def test_rapid_append(self, qtbot) -> None:
        """測試快速追加。"""
        from multall_turbomachinery_design.ui.widgets import ResultText

        text = ResultText("快速追加")
        qtbot.addWidget(text)

        text.set_text("開始\n")
        for i in range(100):
            text.append_text(f"行 {i}\n")

        content = text.get_text()
        assert "開始" in content
        assert "行 99" in content


class TestStyleEdgeCases:
    """樣式系統邊界測試。"""

    def test_repeated_theme_switch(self) -> None:
        """測試重複主題切換。"""
        from multall_turbomachinery_design.ui.styles import StyleManager, ThemeMode

        manager = StyleManager.get_instance()

        # 快速切換主題多次
        for _ in range(10):
            manager.theme = ThemeMode.LIGHT
            assert manager.palette.background == "#FFFFFF"

            manager.theme = ThemeMode.DARK
            assert manager.palette.background == "#1E1E1E"

        # 最終恢復淺色
        manager.theme = ThemeMode.LIGHT

    def test_stylesheet_regeneration(self) -> None:
        """測試樣式表重新生成。"""
        from multall_turbomachinery_design.ui.styles import StyleManager, ThemeMode

        manager = StyleManager.get_instance()
        manager.theme = ThemeMode.LIGHT

        # 多次獲取樣式表
        styles = [manager.get_stylesheet() for _ in range(5)]

        # 所有樣式表應該相同
        assert all(s == styles[0] for s in styles)

    def test_singleton_pattern(self) -> None:
        """測試單例模式。"""
        from multall_turbomachinery_design.ui.styles import StyleManager

        manager1 = StyleManager.get_instance()
        manager2 = StyleManager.get_instance()
        manager3 = StyleManager()

        # 所有實例應該是同一個
        assert manager1 is manager2
        assert manager2 is manager3


class TestDisabledStateEdgeCases:
    """禁用狀態邊界測試。"""

    def test_disabled_value_access(self, qtbot) -> None:
        """測試禁用狀態下的值訪問。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float("test", "測試:", value=100.0)

        # 禁用
        group.set_enabled("test", False)
        assert not spin.isEnabled()

        # 應該仍能讀取值
        assert group.get_value("test") == 100.0

        # 通過代碼設置值應該仍然有效
        group.set_value("test", 200.0)
        assert group.get_value("test") == 200.0

    def test_toggle_enabled_state(self, qtbot) -> None:
        """測試切換啟用狀態。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float("test", "測試:", value=100.0)

        # 多次切換
        for _ in range(5):
            group.set_enabled("test", False)
            assert not spin.isEnabled()

            group.set_enabled("test", True)
            assert spin.isEnabled()
