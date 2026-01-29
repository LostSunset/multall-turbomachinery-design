# -*- coding: utf-8 -*-
"""測試基本匯入。"""

from __future__ import annotations


def test_import_package() -> None:
    """測試主套件匯入。"""
    import multall_turbomachinery_design

    assert multall_turbomachinery_design.__version__ == "0.1.0"


def test_import_modules() -> None:
    """測試各模組匯入。"""
    from multall_turbomachinery_design import meangen, multall, stagen, ui, utils

    assert meangen is not None
    assert stagen is not None
    assert multall is not None
    assert ui is not None
    assert utils is not None


def test_utf8_support() -> None:
    """測試 UTF-8 / 正體中文支援。"""
    test_string = "渦輪機械設計系統"
    assert len(test_string) == 8
    assert test_string.encode("utf-8").decode("utf-8") == test_string
