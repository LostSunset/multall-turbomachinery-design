# -*- coding: utf-8 -*-
"""輸入/輸出處理模組。

提供 meangen.in 讀取和 stagen.dat 生成功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .data_structures import (
    FlowType,
    GasProperties,
    InputType,
    MachineType,
    MeangenConfig,
    RadiusType,
    StageDesign,
)


class MeangenInputReader:
    """MEANGEN 輸入檔案讀取器。"""

    def __init__(self, encoding: str = "utf-8") -> None:
        """初始化輸入讀取器。

        Args:
            encoding: 檔案編碼
        """
        self.encoding = encoding

    def read_config(self, input_file: str | Path) -> MeangenConfig:
        """讀取 meangen.in 設定檔。

        Args:
            input_file: 輸入檔案路徑

        Returns:
            MEANGEN 配置
        """
        with open(input_file, encoding=self.encoding) as f:
            # 讀取機械類型
            turbo_typ = self._read_line(f).strip()
            machine_type = (
                MachineType.TURBINE if turbo_typ == "T" else MachineType.COMPRESSOR
            )

            # 讀取流動類型
            flo_typ = self._read_line(f).strip()
            flow_type = FlowType.AXIAL if flo_typ == "AXI" else FlowType.MIXED

            # 讀取氣體性質
            rgas, gamma = self._read_floats(f, 2)
            poin, toin = self._read_floats(f, 2)

            gas = GasProperties(rgas=rgas, gamma=gamma, poin=poin, toin=toin)

            # 讀取級數
            nstages = self._read_int(f)

            # 讀取設計位置（H=Hub, M=Mid, T=Tip）
            design_location = self._read_line(f).strip()

            # 讀取轉速和質量流量
            rpm = self._read_float(f)
            mass_flow = self._read_float(f)

            # 創建配置
            config = MeangenConfig(
                machine_type=machine_type,
                flow_type=flow_type,
                gas=gas,
                nstages=nstages,
                rpm=rpm,
                mass_flow=mass_flow,
                design_radius=0.0,  # 將在讀取各級時更新
                design_location=design_location,
            )

            # 讀取各級設計
            for istage in range(nstages):
                stage = self._read_stage_design(f, istage + 1, config)
                config.stages.append(stage)

        return config

    def _read_stage_design(
        self, f: TextIO, stage_num: int, config: MeangenConfig
    ) -> StageDesign:
        """讀取單級設計參數。

        Args:
            f: 檔案物件
            stage_num: 級號
            config: MEANGEN 配置

        Returns:
            級設計
        """
        # 讀取輸入類型
        input_type = InputType(self._read_line(f).strip())

        # 根據輸入類型讀取參數
        if input_type == InputType.TYPE_A:
            # 反應度/流量係數/負荷係數
            reaction, phi, psi = self._read_floats(f, 3)
        elif input_type == InputType.TYPE_B:
            # 流量/角度
            phi = self._read_float(f)
            alpha_2, beta_1 = self._read_floats(f, 2)
        else:
            # 其他類型
            phi = psi = reaction = 0.0

        # 讀取半徑類型
        radius_type = RadiusType(self._read_line(f).strip())

        # 讀取設計半徑或焓變
        if radius_type == RadiusType.DIRECT:
            r_design = self._read_float(f)
        else:
            _dho = self._read_float(f)  # 焓變，未來將用於計算半徑
            r_design = 0.5  # 暫時使用默認值

        # 讀取軸向弦長
        axial_chord_1, axial_chord_2 = self._read_floats(f, 2)

        # 讀取間隙
        row_gap, stage_gap = self._read_floats(f, 2)

        # 讀取堵塞因子（v17.4）
        fblock_le, fblock_te = self._read_floats(f, 2)

        # 讀取效率估計
        efficiency = self._read_float(f)

        # 讀取偏角估計
        devn1, devn2 = self._read_floats(f, 2)

        # 讀取入射角
        ainc1, ainc2 = self._read_floats(f, 2)

        # 讀取扭轉比例（v17.4）
        frac_twist = self._read_float(f)

        # 讀取旋轉選項（v17.4）
        _if_rot = self._read_line(f).strip().lower()  # 旋轉選項，未來將用於自由渦設計

        # TODO: 讀取更多參數...

        # 創建級設計
        stage = StageDesign(
            stage_number=stage_num,
            phi=phi,
            psi=psi,
            reaction=reaction,
            r_design=r_design,
            efficiency=efficiency,
            fblock_le=fblock_le,
            fblock_te=fblock_te,
            frac_twist=frac_twist,
        )

        return stage

    def _read_line(self, f: TextIO) -> str:
        """讀取一行，跳過註釋。

        Args:
            f: 檔案物件

        Returns:
            行內容
        """
        while True:
            line = f.readline()
            if not line:
                return ""
            # 移除註釋（假設註釋以 ! 或 # 開始）
            if "!" in line:
                line = line.split("!")[0]
            if "#" in line:
                line = line.split("#")[0]
            line = line.strip()
            if line:
                return line

    def _read_float(self, f: TextIO) -> float:
        """讀取單個浮點數。

        Args:
            f: 檔案物件

        Returns:
            浮點數
        """
        line = self._read_line(f)
        return float(line.split()[0])

    def _read_int(self, f: TextIO) -> int:
        """讀取單個整數。

        Args:
            f: 檔案物件

        Returns:
            整數
        """
        line = self._read_line(f)
        return int(line.split()[0])

    def _read_floats(self, f: TextIO, n: int) -> list[float]:
        """讀取多個浮點數。

        Args:
            f: 檔案物件
            n: 數量

        Returns:
            浮點數列表
        """
        line = self._read_line(f)
        values = [float(x) for x in line.split()[:n]]
        return values


class StagenOutputWriter:
    """STAGEN 輸出檔案寫入器。"""

    def __init__(self, encoding: str = "utf-8") -> None:
        """初始化輸出寫入器。

        Args:
            encoding: 檔案編碼
        """
        self.encoding = encoding

    def write_stagen_file(
        self, config: MeangenConfig, output_file: str | Path
    ) -> None:
        """寫入 stagen.dat 檔案。

        Args:
            config: MEANGEN 配置
            output_file: 輸出檔案路徑
        """
        with open(output_file, "w", encoding=self.encoding) as f:
            # 寫入標頭
            self._write_header(f, config)

            # 寫入各葉片排
            for stage in config.stages:
                if stage.rotor:
                    self._write_blade_row(f, stage.rotor, stage)
                if stage.stator:
                    self._write_blade_row(f, stage.stator, stage)

            # 寫入邊界條件
            self._write_boundary_conditions(f, config)

    def _write_header(self, f: TextIO, config: MeangenConfig) -> None:
        """寫入檔案標頭。

        Args:
            f: 檔案物件
            config: MEANGEN 配置
        """
        # 第一行：RGAS, GAMMA
        f.write(f"{config.gas.rgas:12.4f}  {config.gas.gamma:8.4f}              ")
        f.write("GAS CONSTANT, GAMMA\n")

        # 第二行：IM, KM
        f.write(f"{config.im:8d}{config.km:8d}                  IM, KM\n")

        # 第三行：FPRAT, FPMAX
        f.write("0.0500   1.5000              FPRAT, FPMAX\n")

        # 第四行：FRRAT, FRMAX
        f.write("0.0500   1.5000              FRRAT, FRMAX\n")

        # 第五行：IFDEFAULTS
        f.write("         0                   IFDEFAULTS\n")

        # 第六行：NROWS, NSECTIONS
        nrows = len(config.stages) * 2  # 每級2排（轉子+定子）
        f.write(f"{nrows:8d}{config.nosect:8d}             NROWS, N SECTIONS\n")

        # 第七行：縮放因子
        f.write("       1.000                 SCALING FACTOR\n")
        f.write("\n")

    def _write_blade_row(
        self, f: TextIO, blade_row: object, stage: StageDesign
    ) -> None:
        """寫入葉片排數據。

        Args:
            f: 檔案物件
            blade_row: 葉片排
            stage: 級設計
        """
        # TODO: 實現完整的葉片排數據寫入
        f.write("***STARTING DATA FOR A NEW BLADE ROW***\n")
        f.write(f"  BLADE ROW NUMBER = {stage.stage_number}\n")
        # ... 更多數據

    def _write_boundary_conditions(self, f: TextIO, config: MeangenConfig) -> None:
        """寫入邊界條件。

        Args:
            f: 檔案物件
            config: MEANGEN 配置
        """
        # TODO: 實現邊界條件寫入
        pass


class MeangenOutputWriter:
    """MEANGEN.OUT 輸出檔案寫入器。"""

    def __init__(self, encoding: str = "utf-8") -> None:
        """初始化輸出寫入器。

        Args:
            encoding: 檔案編碼
        """
        self.encoding = encoding

    def write_output_file(
        self, config: MeangenConfig, output_file: str | Path
    ) -> None:
        """寫入 meangen.out 檔案。

        Args:
            config: MEANGEN 配置
            output_file: 輸出檔案路徑
        """
        with open(output_file, "w", encoding=self.encoding) as f:
            # 寫入配置的鏡像副本，格式化以便編輯
            f.write("MULTALL MEANGEN OUTPUT\n")
            f.write("=" * 50 + "\n\n")

            # 機械類型
            f.write(f"{config.machine_type.value:20s}  MACHINE TYPE (T/C)\n")

            # 流動類型
            f.write(f"{config.flow_type.value:20s}  FLOW TYPE (AXI/MIX)\n")

            # 氣體性質
            f.write(f"{config.gas.rgas:20.4f}  GAS CONSTANT [J/(kg·K)]\n")
            f.write(f"{config.gas.gamma:20.4f}  GAMMA\n")
            f.write(f"{config.gas.poin:20.4f}  INLET TOTAL PRESSURE [Bar]\n")
            f.write(f"{config.gas.toin:20.4f}  INLET TOTAL TEMPERATURE [K]\n")

            # 運行參數
            f.write(f"{config.nstages:20d}  NUMBER OF STAGES\n")
            f.write(f"{config.rpm:20.2f}  RPM\n")
            f.write(f"{config.mass_flow:20.4f}  MASS FLOW [kg/s]\n")

            f.write("\n")
            f.write("STAGE DESIGNS\n")
            f.write("-" * 50 + "\n")

            # 各級參數
            for stage in config.stages:
                f.write(f"\nSTAGE {stage.stage_number}\n")
                f.write(f"  Flow Coefficient:    {stage.phi:10.4f}\n")
                f.write(f"  Loading Coefficient: {stage.psi:10.4f}\n")
                f.write(f"  Reaction:            {stage.reaction:10.4f}\n")
                f.write(f"  Design Radius:       {stage.r_design:10.4f} m\n")
                f.write(f"  Efficiency:          {stage.efficiency:10.4f}\n")
