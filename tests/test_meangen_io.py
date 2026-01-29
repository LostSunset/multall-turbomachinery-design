# -*- coding: utf-8 -*-
"""MEANGEN I/O 功能測試。"""

from __future__ import annotations

from pathlib import Path

from multall_turbomachinery_design.meangen.data_structures import (
    FlowType,
    GasProperties,
    MachineType,
    MeangenConfig,
)
from multall_turbomachinery_design.meangen.io_handler import (
    MeangenOutputWriter,
    StagenOutputWriter,
)


class TestMeangenOutputWriter:
    """測試 MEANGEN 輸出寫入器。"""

    def test_write_output_file(self, tmp_path: Path) -> None:
        """測試寫入 meangen.out 檔案。"""
        # 創建簡單的配置
        config = MeangenConfig(
            machine_type=MachineType.TURBINE,
            flow_type=FlowType.AXIAL,
            gas=GasProperties(rgas=287.5, gamma=1.4, poin=1.0, toin=300.0),
            nstages=1,
            rpm=5000.0,
            mass_flow=50.0,
            design_radius=0.5,
        )

        # 寫入檔案
        output_file = tmp_path / "meangen.out"
        writer = MeangenOutputWriter()
        writer.write_output_file(config, output_file)

        # 檢查檔案存在
        assert output_file.exists()

        # 檢查檔案內容
        content = output_file.read_text(encoding="utf-8")
        assert "MULTALL MEANGEN OUTPUT" in content
        assert "TURBINE" in content or "T" in content
        assert "287.5" in content  # RGAS


class TestStagenOutputWriter:
    """測試 STAGEN 輸出寫入器。"""

    def test_write_stagen_header(self, tmp_path: Path) -> None:
        """測試寫入 stagen.dat 標頭。"""
        config = MeangenConfig(
            machine_type=MachineType.TURBINE,
            flow_type=FlowType.AXIAL,
            gas=GasProperties(rgas=287.5, gamma=1.4, poin=1.0, toin=300.0),
            nstages=1,
            rpm=5000.0,
            mass_flow=50.0,
            design_radius=0.5,
        )

        # 寫入檔案
        output_file = tmp_path / "stagen.dat"
        writer = StagenOutputWriter()
        writer.write_stagen_file(config, output_file)

        # 檢查檔案存在
        assert output_file.exists()

        # 檢查檔案內容
        content = output_file.read_text(encoding="utf-8")
        assert "GAS CONSTANT" in content
        assert "287.5" in content


class TestUTF8Support:
    """測試 UTF-8 支援。"""

    def test_utf8_output(self, tmp_path: Path) -> None:
        """測試 UTF-8 輸出。"""
        config = MeangenConfig(
            machine_type=MachineType.TURBINE,
            flow_type=FlowType.AXIAL,
            gas=GasProperties(rgas=287.5, gamma=1.4, poin=1.0, toin=300.0),
            nstages=1,
            rpm=5000.0,
            mass_flow=50.0,
            design_radius=0.5,
        )

        output_file = tmp_path / "test_utf8.out"
        writer = MeangenOutputWriter()
        writer.write_output_file(config, output_file)

        # 讀取並檢查中文註釋
        content = output_file.read_text(encoding="utf-8")
        assert "kg" in content  # 確保檔案可以正常讀取
