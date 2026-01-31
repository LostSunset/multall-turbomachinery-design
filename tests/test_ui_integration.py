# -*- coding: utf-8 -*-
"""UI 整合測試。

測試元件之間的連動邏輯和資料流。
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


class TestParameterGroupSignals:
    """測試 ParameterGroup 的信號連動。"""

    def test_parameter_changed_signal(self, qtbot) -> None:
        """測試參數變更信號。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        spin = group.add_float("temp", "溫度:", value=300.0)

        # 使用 qtbot 捕獲信號
        with qtbot.waitSignal(group.parameterChanged, timeout=1000) as blocker:
            spin.setValue(350.0)

        assert blocker.args[0] == "temp"
        assert blocker.args[1] == 350.0

    def test_combo_changed_signal(self, qtbot) -> None:
        """測試下拉選擇變更信號。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        combo = group.add_combo("type", "類型:", ["A", "B", "C"], current=0)

        with qtbot.waitSignal(group.parameterChanged, timeout=1000) as blocker:
            combo.setCurrentIndex(2)

        assert blocker.args[0] == "type"
        assert blocker.args[1] == 2


class TestParameterFormSignals:
    """測試 ParameterForm 的信號連動。"""

    def test_nested_parameter_changed_signal(self, qtbot) -> None:
        """測試嵌套參數變更信號。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group = form.add_group("inlet", "入口條件")
        spin = group.add_float("pressure", "壓力:", value=101325.0)

        with qtbot.waitSignal(form.parameterChanged, timeout=1000) as blocker:
            spin.setValue(200000.0)

        assert blocker.args[0] == "inlet"
        assert blocker.args[1] == "pressure"
        assert blocker.args[2] == 200000.0


class TestDataFlowIntegration:
    """測試資料流整合。"""

    def test_form_to_values_round_trip(self, qtbot) -> None:
        """測試表單資料的往返處理。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        # 設置初始值
        group1 = form.add_group("basic", "基本")
        group1.add_float("value1", "值1:", value=100.0)
        group1.add_int("value2", "值2:", value=50)

        group2 = form.add_group("advanced", "進階")
        group2.add_float("value3", "值3:", value=0.5)

        # 取得所有值
        values = form.get_all_values()

        # 驗證資料完整性
        assert values["basic"]["value1"] == 100.0
        assert values["basic"]["value2"] == 50
        assert values["advanced"]["value3"] == 0.5

        # 修改值
        group1.set_value("value1", 200.0)
        values = form.get_all_values()
        assert values["basic"]["value1"] == 200.0

    def test_table_data_integrity(self, qtbot) -> None:
        """測試表格資料完整性。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        # 設置標頭和資料
        headers = ["參數", "設計值", "實際值", "偏差"]
        table.set_headers(headers)

        data = [
            ["效率", "0.85", "0.83", "-2.4%"],
            ["功率", "1000", "980", "-2.0%"],
            ["流量", "10.0", "10.2", "+2.0%"],
        ]
        table.set_data(data)

        assert table.get_column_count() == 4
        assert table.get_row_count() == 3


class TestEnableDisableIntegration:
    """測試啟用/禁用狀態的連動。"""

    def test_dependent_parameters(self, qtbot) -> None:
        """測試相依參數的啟用狀態。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 添加主控參數和相依參數
        combo = group.add_combo("mode", "模式:", ["簡單", "進階"], current=0)
        spin = group.add_float("advanced_param", "進階參數:", value=1.0)

        # 模擬依據模式啟用/禁用參數
        def update_enabled():
            is_advanced = combo.currentIndex() == 1
            group.set_enabled("advanced_param", is_advanced)

        combo.currentIndexChanged.connect(update_enabled)
        update_enabled()

        # 初始狀態：簡單模式，進階參數禁用
        assert not spin.isEnabled()

        # 切換到進階模式
        combo.setCurrentIndex(1)
        assert spin.isEnabled()

        # 切換回簡單模式
        combo.setCurrentIndex(0)
        assert not spin.isEnabled()


class TestStyleIntegration:
    """測試樣式系統整合。"""

    def test_style_constants(self) -> None:
        """測試樣式常數。"""
        from multall_turbomachinery_design.ui.styles import (
            BORDER_RADIUS_MD,
            BORDER_RADIUS_SM,
            GRID_UNIT,
            MIN_TOUCH_SIZE,
        )

        assert GRID_UNIT == 8
        assert MIN_TOUCH_SIZE == 44
        assert BORDER_RADIUS_SM == 4
        assert BORDER_RADIUS_MD == 8

    def test_spacing_system(self) -> None:
        """測試間距系統。"""
        from multall_turbomachinery_design.ui.styles import Spacing

        sp = Spacing()
        assert sp.xs == 4
        assert sp.sm == 8
        assert sp.md == 16
        assert sp.lg == 24
        assert sp.xl == 32
        assert sp.xxl == 48

    def test_typography_system(self) -> None:
        """測試字型系統。"""
        from multall_turbomachinery_design.ui.styles import Typography

        ty = Typography()
        assert ty.h1 > ty.h2 > ty.h3 > ty.h4 > ty.h5 > ty.h6
        assert ty.body == 13
        assert ty.body_sm == 12
        assert ty.caption == 11

    def test_theme_switching(self) -> None:
        """測試主題切換。"""
        from multall_turbomachinery_design.ui.styles import (
            ColorPalette,
            DarkColorPalette,
            StyleManager,
            ThemeMode,
        )

        manager = StyleManager.get_instance()

        # 測試淺色主題
        manager.theme = ThemeMode.LIGHT
        assert isinstance(manager.palette, ColorPalette)
        assert manager.palette.background == "#FFFFFF"

        # 測試深色主題
        manager.theme = ThemeMode.DARK
        assert isinstance(manager.palette, DarkColorPalette)
        assert manager.palette.background == "#1E1E1E"

        # 恢復淺色主題
        manager.theme = ThemeMode.LIGHT

    def test_stylesheet_generation(self) -> None:
        """測試樣式表生成。"""
        from multall_turbomachinery_design.ui.styles import StyleManager

        manager = StyleManager.get_instance()
        stylesheet = manager.get_stylesheet()

        # 驗證樣式表包含關鍵元素
        assert "QWidget" in stylesheet
        assert "QPushButton" in stylesheet
        assert "QSpinBox" in stylesheet
        assert "QComboBox" in stylesheet
        assert "font-family" in stylesheet


class TestValidationIntegration:
    """測試驗證整合。"""

    def test_group_validation(self, qtbot) -> None:
        """測試群組驗證。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        group.add_float("a", "A:", minimum=0.0, maximum=100.0, value=50.0)
        group.add_int("b", "B:", minimum=1, maximum=10, value=5)

        valid, errors = group.validate()
        assert valid
        assert len(errors) == 0

    def test_form_validation(self, qtbot) -> None:
        """測試表單驗證。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        group1 = form.add_group("g1", "群組1")
        group1.add_float("v1", "V1:", minimum=0, maximum=100, value=50)

        group2 = form.add_group("g2", "群組2")
        group2.add_int("v2", "V2:", minimum=1, maximum=10, value=5)

        valid, errors = form.validate_all()
        assert valid
        assert len(errors) == 0
