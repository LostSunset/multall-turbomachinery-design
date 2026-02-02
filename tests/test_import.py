# -*- coding: utf-8 -*-
"""測試基本匯入。"""

from __future__ import annotations

import pytest


def test_import_package() -> None:
    """測試主套件匯入。"""
    import multall_turbomachinery_design

    assert multall_turbomachinery_design.__version__ == "0.1.0"


def test_import_core_modules() -> None:
    """測試核心模組匯入（不含 UI）。"""
    from multall_turbomachinery_design import meangen, multall, stagen, utils

    assert meangen is not None
    assert stagen is not None
    assert multall is not None
    assert utils is not None


def test_import_ui_module() -> None:
    """測試 UI 模組匯入。"""
    try:
        from multall_turbomachinery_design import ui

        assert ui is not None
    except ImportError as e:
        # 在無顯示環境中（如 CI）跳過
        pytest.skip(f"UI 模組無法匯入（可能缺少顯示庫）: {e}")


def test_utf8_support() -> None:
    """測試 UTF-8 / 正體中文支援。"""
    test_string = "渦輪機械設計系統"
    assert len(test_string) == 8
    assert test_string.encode("utf-8").decode("utf-8") == test_string
