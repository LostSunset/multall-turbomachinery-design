# -*- coding: utf-8 -*-
"""命令行介面測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCLIParser:
    """CLI 解析器測試。"""

    def test_create_parser(self) -> None:
        """測試創建解析器。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        assert parser is not None
        assert parser.prog == "multall"

    def test_parse_version(self) -> None:
        """測試版本參數。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_parse_help(self) -> None:
        """測試幫助參數。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_parse_info(self) -> None:
        """測試 info 命令解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_parse_info_check_deps(self) -> None:
        """測試 info --check-deps 解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["info", "--check-deps"])
        assert args.command == "info"
        assert args.check_deps is True

    def test_parse_meangen(self, tmp_path: Path) -> None:
        """測試 meangen 命令解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        input_file = tmp_path / "test.json"
        args = parser.parse_args(["meangen", str(input_file)])
        assert args.command == "meangen"
        assert args.input == input_file

    def test_parse_meangen_with_output(self, tmp_path: Path) -> None:
        """測試 meangen -o 解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        input_file = tmp_path / "test.json"
        output_dir = tmp_path / "output"
        args = parser.parse_args(["meangen", str(input_file), "-o", str(output_dir)])
        assert args.output == output_dir

    def test_parse_gui(self) -> None:
        """測試 gui 命令解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["gui"])
        assert args.command == "gui"

    def test_parse_validate(self, tmp_path: Path) -> None:
        """測試 validate 命令解析。"""
        from multall_turbomachinery_design.cli.main import create_parser

        parser = create_parser()
        input_file = tmp_path / "test.json"
        args = parser.parse_args(["validate", str(input_file)])
        assert args.command == "validate"
        assert args.input == input_file


class TestCLICommands:
    """CLI 命令執行測試。"""

    def test_run_info(self, capsys) -> None:
        """測試 info 命令執行。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["info"])
        assert result == 0

        captured = capsys.readouterr()
        assert "MULTALL" in captured.out
        assert "版本" in captured.out

    def test_run_info_check_deps(self, capsys) -> None:
        """測試 info --check-deps 執行。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["info", "--check-deps"])
        assert result == 0

        captured = capsys.readouterr()
        assert "numpy" in captured.out
        assert "matplotlib" in captured.out

    def test_run_validate_valid_file(self, tmp_path: Path, capsys) -> None:
        """測試 validate 有效文件。"""
        from multall_turbomachinery_design.cli.main import app

        # 創建有效配置
        config_file = tmp_path / "valid.json"
        config = {
            "machine_type": "TURBINE",
            "flow_type": "AXIAL",
            "stages": [{"stage_number": 1}],
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = app(["validate", str(config_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "驗證通過" in captured.out

    def test_run_validate_missing_fields(self, tmp_path: Path, capsys) -> None:
        """測試 validate 缺少欄位。"""
        from multall_turbomachinery_design.cli.main import app

        # 創建缺少欄位的配置
        config_file = tmp_path / "missing.json"
        config = {"rpm": 10000}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = app(["validate", str(config_file)])
        assert result == 0  # 警告不算錯誤

        captured = capsys.readouterr()
        assert "警告" in captured.out

    def test_run_validate_invalid_json(self, tmp_path: Path, capsys) -> None:
        """測試 validate 無效 JSON。"""
        from multall_turbomachinery_design.cli.main import app

        # 創建無效 JSON
        config_file = tmp_path / "invalid.json"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        result = app(["validate", str(config_file)])
        assert result == 1

        captured = capsys.readouterr()
        assert "JSON 格式錯誤" in captured.out

    def test_run_validate_nonexistent_file(self, tmp_path: Path, capsys) -> None:
        """測試 validate 不存在的文件。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["validate", str(tmp_path / "nonexistent.json")])
        assert result == 1

        captured = capsys.readouterr()
        assert "文件不存在" in captured.out

    def test_run_meangen_nonexistent_file(self, tmp_path: Path, capsys) -> None:
        """測試 meangen 不存在的文件。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["meangen", str(tmp_path / "nonexistent.json")])
        assert result == 1

        captured = capsys.readouterr()
        assert "不存在" in captured.out

    def test_run_no_command(self, capsys) -> None:
        """測試沒有命令。"""
        from multall_turbomachinery_design.cli.main import app

        result = app([])
        assert result == 0  # 顯示幫助

    def test_app_main_function(self) -> None:
        """測試 app 函數。"""
        from multall_turbomachinery_design.cli.main import app

        # 測試 info 命令
        result = app(["info"])
        assert result == 0


class TestCLIPlot:
    """CLI 繪圖命令測試。"""

    def test_plot_no_type(self, capsys) -> None:
        """測試 plot 沒有類型。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["plot"])
        assert result == 1

        captured = capsys.readouterr()
        assert "請指定圖表類型" in captured.out

    def test_plot_nonexistent_file(self, tmp_path: Path, capsys) -> None:
        """測試 plot 不存在的文件。"""
        from multall_turbomachinery_design.cli.main import app

        result = app(["plot", "convergence", str(tmp_path / "nonexistent.csv")])
        assert result == 1

        captured = capsys.readouterr()
        assert "不存在" in captured.out


class TestCLIModule:
    """CLI 模組測試。"""

    def test_import_cli(self) -> None:
        """測試導入 CLI 模組。"""
        from multall_turbomachinery_design import cli

        assert cli is not None

    def test_import_app(self) -> None:
        """測試導入 app 函數。"""
        from multall_turbomachinery_design.cli import app

        assert app is not None
        assert callable(app)

    def test_import_main(self) -> None:
        """測試導入 main 函數。"""
        from multall_turbomachinery_design.cli import main

        assert main is not None
        assert callable(main)
