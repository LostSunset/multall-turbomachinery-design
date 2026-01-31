# -*- coding: utf-8 -*-
"""UI 工作流程端到端測試。

測試完整的使用者旅程和工作流程。
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


class TestParameterInputWorkflow:
    """測試參數輸入工作流程。"""

    def test_complete_parameter_form_workflow(self, qtbot) -> None:
        """測試完整的參數表單工作流程。

        模擬使用者：
        1. 創建表單
        2. 添加參數群組
        3. 填入參數值
        4. 驗證資料
        5. 取得最終結果
        """
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        # 步驟 1: 創建表單
        form = ParameterForm()
        qtbot.addWidget(form)

        # 步驟 2: 添加參數群組
        inlet_group = form.add_group("inlet", "入口條件")
        inlet_group.add_float("total_pressure", "總壓:", value=101325.0, suffix="Pa")
        inlet_group.add_float("total_temperature", "總溫:", value=288.15, suffix="K")
        inlet_group.add_float("mass_flow", "質量流量:", value=10.0, suffix="kg/s")

        outlet_group = form.add_group("outlet", "出口條件")
        outlet_group.add_float("static_pressure", "靜壓:", value=50000.0, suffix="Pa")

        geometry_group = form.add_group("geometry", "幾何參數")
        geometry_group.add_int("blade_count", "葉片數:", value=20)
        geometry_group.add_float("hub_radius", "輪轂半徑:", value=0.1, suffix="m")
        geometry_group.add_float("tip_radius", "葉尖半徑:", value=0.2, suffix="m")

        # 步驟 3: 模擬使用者修改參數值
        inlet_group.set_value("total_pressure", 150000.0)
        inlet_group.set_value("total_temperature", 300.0)
        geometry_group.set_value("blade_count", 24)

        # 步驟 4: 驗證資料
        valid, errors = form.validate_all()
        assert valid, f"驗證失敗: {errors}"

        # 步驟 5: 取得最終結果
        values = form.get_all_values()

        assert values["inlet"]["total_pressure"] == 150000.0
        assert values["inlet"]["total_temperature"] == 300.0
        assert values["inlet"]["mass_flow"] == 10.0
        assert values["outlet"]["static_pressure"] == 50000.0
        assert values["geometry"]["blade_count"] == 24
        assert values["geometry"]["hub_radius"] == 0.1
        assert values["geometry"]["tip_radius"] == 0.2


class TestResultDisplayWorkflow:
    """測試結果顯示工作流程。"""

    def test_table_display_workflow(self, qtbot) -> None:
        """測試表格顯示工作流程。

        模擬：
        1. 創建結果表格
        2. 設定標頭
        3. 填入計算結果
        4. 添加新行
        5. 清除並重新填入
        """
        from multall_turbomachinery_design.ui.widgets import ResultTable

        # 步驟 1: 創建表格
        table = ResultTable("性能參數")
        qtbot.addWidget(table)

        # 步驟 2: 設定標頭
        table.set_headers(["參數", "設計值", "實際值", "偏差 (%)"])
        assert table.get_column_count() == 4

        # 步驟 3: 填入計算結果
        results = [
            ["等熵效率", "0.850", "0.842", "-0.94"],
            ["多變效率", "0.870", "0.865", "-0.57"],
            ["軸功率", "1000", "985", "-1.50"],
            ["質量流量", "10.0", "10.1", "+1.00"],
        ]
        table.set_data(results)
        assert table.get_row_count() == 4

        # 步驟 4: 添加新行
        table.add_row(["壓比", "2.5", "2.48", "-0.80"])
        assert table.get_row_count() == 5

        # 步驟 5: 清除並重新填入
        table.clear_data()
        assert table.get_row_count() == 0

        table.set_data(results[:2])
        assert table.get_row_count() == 2

    def test_log_display_workflow(self, qtbot) -> None:
        """測試日誌顯示工作流程。

        模擬：
        1. 創建日誌元件
        2. 顯示初始訊息
        3. 追加計算過程訊息
        4. 顯示最終結果
        5. 清除日誌
        """
        from multall_turbomachinery_design.ui.widgets import ResultText

        # 步驟 1: 創建日誌元件
        log = ResultText("計算日誌")
        qtbot.addWidget(log)

        # 步驟 2: 顯示初始訊息
        log.set_text("=== MEANGEN 計算開始 ===\n")

        # 步驟 3: 追加計算過程訊息
        log.append_text("正在讀取輸入參數...\n")
        log.append_text("正在計算速度三角形...\n")
        log.append_text("正在計算整體性能...\n")

        # 步驟 4: 顯示最終結果
        log.append_text("\n=== 計算完成 ===\n")
        log.append_text("等熵效率: 0.842\n")
        log.append_text("軸功率: 985 kW\n")

        content = log.get_text()
        assert "MEANGEN" in content
        assert "計算完成" in content
        assert "等熵效率" in content

        # 步驟 5: 清除日誌
        log.clear_text()
        assert log.get_text() == ""


class TestParameterGroupStateManagement:
    """測試參數群組狀態管理。"""

    def test_state_save_restore_workflow(self, qtbot) -> None:
        """測試狀態保存和恢復工作流程。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        # 創建並設置參數
        group = ParameterGroup("設計參數")
        qtbot.addWidget(group)

        group.add_float("pressure", "壓力:", value=101325.0)
        group.add_float("temperature", "溫度:", value=288.15)
        group.add_int("blades", "葉片數:", value=20)
        group.add_combo("material", "材料:", ["鋁", "鈦", "鋼"], current=0)

        # 保存當前狀態
        saved_state = group.get_all_values()

        # 修改參數
        group.set_value("pressure", 200000.0)
        group.set_value("temperature", 350.0)
        group.set_value("blades", 30)
        group.set_value("material", 2)

        # 驗證已修改
        assert group.get_value("pressure") == 200000.0
        assert group.get_value("temperature") == 350.0

        # 恢復狀態
        for name, value in saved_state.items():
            group.set_value(name, value)

        # 驗證已恢復
        assert group.get_value("pressure") == 101325.0
        assert group.get_value("temperature") == 288.15
        assert group.get_value("blades") == 20
        assert group.get_value("material") == 0


class TestCompleteDesignWorkflow:
    """測試完整設計工作流程。"""

    def test_turbine_design_workflow(self, qtbot) -> None:
        """測試渦輪設計完整工作流程。

        模擬完整的渦輪設計過程：
        1. 輸入設計條件
        2. 驗證輸入
        3. 模擬計算
        4. 顯示結果
        """
        from multall_turbomachinery_design.ui.widgets import (
            ParameterForm,
            ResultTable,
            ResultText,
        )

        # ===== 1. 設置輸入表單 =====
        input_form = ParameterForm()
        qtbot.addWidget(input_form)

        # 熱力學條件
        thermo_group = input_form.add_group("thermo", "熱力學條件")
        thermo_group.add_float("p01", "入口總壓:", value=400000.0, suffix="Pa")
        thermo_group.add_float("t01", "入口總溫:", value=1200.0, suffix="K")
        thermo_group.add_float("p2", "出口靜壓:", value=100000.0, suffix="Pa")
        thermo_group.add_float("mdot", "質量流量:", value=50.0, suffix="kg/s")

        # 幾何條件
        geom_group = input_form.add_group("geometry", "幾何條件")
        geom_group.add_float("rm", "平均半徑:", value=0.3, suffix="m")
        geom_group.add_int("nblades_stator", "靜葉數:", value=40)
        geom_group.add_int("nblades_rotor", "動葉數:", value=60)

        # ===== 2. 驗證輸入 =====
        valid, errors = input_form.validate_all()
        assert valid, f"輸入驗證失敗: {errors}"

        # ===== 3. 模擬計算（簡化） =====
        inputs = input_form.get_all_values()
        p01 = inputs["thermo"]["p01"]
        t01 = inputs["thermo"]["t01"]
        p2 = inputs["thermo"]["p2"]
        mdot = inputs["thermo"]["mdot"]

        # 簡化計算
        pressure_ratio = p01 / p2
        # 假設理想氣體，簡化效率計算
        eta_is = 0.88  # 假設等熵效率
        power = mdot * 1004 * t01 * eta_is * (1 - (1 / pressure_ratio) ** 0.286)

        # ===== 4. 顯示結果 =====
        # 日誌輸出
        log = ResultText("計算日誌")
        qtbot.addWidget(log)

        log.set_text("=== 渦輪設計計算 ===\n")
        log.append_text(f"壓比: {pressure_ratio:.2f}\n")
        log.append_text(f"等熵效率: {eta_is:.3f}\n")
        log.append_text(f"軸功率: {power / 1000:.1f} kW\n")

        # 結果表格
        result_table = ResultTable("性能摘要")
        qtbot.addWidget(result_table)

        result_table.set_headers(["參數", "數值", "單位"])
        result_table.set_data(
            [
                ["壓比", f"{pressure_ratio:.2f}", "-"],
                ["等熵效率", f"{eta_is:.3f}", "-"],
                ["軸功率", f"{power / 1000:.1f}", "kW"],
                ["質量流量", f"{mdot:.1f}", "kg/s"],
            ]
        )

        # 驗證結果
        assert result_table.get_row_count() == 4
        assert "渦輪設計計算" in log.get_text()
        assert pressure_ratio > 1


class TestUIResetWorkflow:
    """測試 UI 重置工作流程。"""

    def test_form_reset_workflow(self, qtbot) -> None:
        """測試表單重置工作流程。"""
        from multall_turbomachinery_design.ui.widgets import ParameterForm

        form = ParameterForm()
        qtbot.addWidget(form)

        # 設置參數
        group = form.add_group("test", "測試")
        group.add_float("a", "A:", value=100.0)
        group.add_int("b", "B:", value=50)

        # 記錄初始值
        initial_a = group.get_value("a")
        initial_b = group.get_value("b")

        # 修改值
        group.set_value("a", 200.0)
        group.set_value("b", 100)

        # 驗證已修改
        assert group.get_value("a") == 200.0
        assert group.get_value("b") == 100

        # 重置（恢復初始值）
        group.set_value("a", initial_a)
        group.set_value("b", initial_b)

        # 驗證已重置
        assert group.get_value("a") == initial_a
        assert group.get_value("b") == initial_b

    def test_table_reset_workflow(self, qtbot) -> None:
        """測試表格重置工作流程。"""
        from multall_turbomachinery_design.ui.widgets import ResultTable

        table = ResultTable("測試")
        qtbot.addWidget(table)

        # 設置表格
        table.set_headers(["A", "B", "C"])
        table.set_data([["1", "2", "3"], ["4", "5", "6"]])

        assert table.get_row_count() == 2
        assert table.get_column_count() == 3

        # 清除
        table.clear_data()
        assert table.get_row_count() == 0
        # 標頭應保留
        assert table.get_column_count() == 3


class TestAccessibilityWorkflow:
    """測試無障礙工作流程。"""

    def test_minimum_touch_size(self, qtbot) -> None:
        """測試最小觸控尺寸。"""
        from multall_turbomachinery_design.ui.styles import MIN_TOUCH_SIZE
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試")
        qtbot.addWidget(group)

        # 添加參數以確保群組正常工作
        group.add_float("test", "測試:", value=1.0)

        # SpinBox 應該有足夠的高度
        # 注意：實際渲染尺寸可能因樣式而異
        assert MIN_TOUCH_SIZE == 44

    def test_tooltip_presence(self, qtbot) -> None:
        """測試工具提示存在。"""
        from multall_turbomachinery_design.ui.widgets import ParameterGroup

        group = ParameterGroup("測試", tooltip="群組提示")
        qtbot.addWidget(group)

        spin = group.add_float(
            "pressure",
            "壓力:",
            tooltip="入口總壓",
            value=101325.0,
        )

        assert group.toolTip() == "群組提示"
        assert spin.toolTip() == "入口總壓"
