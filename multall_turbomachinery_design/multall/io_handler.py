# -*- coding: utf-8 -*-
"""MULTALL I/O 處理器。

讀取 MULTALL 輸入文件，寫入結果輸出文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .data_structures import (
    GasProperties,
    GasType,
    MultallConfig,
    TimeStepType,
    ViscousModel,
)


class MultallInputReader:
    """MULTALL 輸入文件讀取器。"""

    def __init__(self) -> None:
        """初始化讀取器。"""
        self._file: TextIO | None = None
        self._line_number = 0

    def _read_line(self) -> str:
        """讀取下一行（跳過空行和註釋）。"""
        if self._file is None:
            return ""
        while True:
            line = self._file.readline()
            self._line_number += 1
            if not line:
                return ""
            line = line.strip()
            # 跳過空行和註釋（以 C 開頭）
            if line and not line.upper().startswith("C"):
                return line
        return ""

    def _read_float(self) -> float:
        """讀取單個浮點數。"""
        line = self._read_line()
        return float(line.split()[0]) if line else 0.0

    def _read_int(self) -> int:
        """讀取單個整數。"""
        line = self._read_line()
        return int(line.split()[0]) if line else 0

    def _read_floats(self, n: int) -> list[float]:
        """讀取多個浮點數。"""
        values: list[float] = []
        while len(values) < n:
            line = self._read_line()
            if not line:
                break
            parts = line.split()
            for part in parts:
                if len(values) < n:
                    values.append(float(part))
        return values

    def _read_ints(self, n: int) -> list[int]:
        """讀取多個整數。"""
        values: list[int] = []
        while len(values) < n:
            line = self._read_line()
            if not line:
                break
            parts = line.split()
            for part in parts:
                if len(values) < n:
                    values.append(int(part))
        return values

    def _skip_comment_line(self) -> None:
        """跳過註釋行（DUMMY_INPUT）。"""
        self._read_line()

    def read(self, input_file: str | Path) -> MultallConfig:
        """讀取輸入文件。

        Args:
            input_file: 輸入文件路徑

        Returns:
            MULTALL 配置
        """
        config = MultallConfig()

        with open(input_file, encoding="utf-8", errors="ignore") as f:
            self._file = f
            self._line_number = 0

            # 讀取標題
            config.title = self._read_line()

            # 讀取氣體性質
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            cp = float(parts[0]) if len(parts) > 0 else 1005.0
            gamma = float(parts[1]) if len(parts) > 1 else 1.4
            ifgas = int(parts[2]) if len(parts) > 2 else 0

            config.gas = GasProperties(
                cp=abs(cp),
                gamma=gamma,
                gas_type=GasType(ifgas),
            )

            if cp < 0 or ifgas == 1:
                # 變 CP 氣體
                self._skip_comment_line()
                line = self._read_line()
                parts = line.split()
                config.gas.cp1 = float(parts[0]) if len(parts) > 0 else 1272.5
                config.gas.cp2 = float(parts[1]) if len(parts) > 1 else 0.2125
                config.gas.cp3 = float(parts[2]) if len(parts) > 2 else 0.000015625
                config.gas.tref = float(parts[3]) if len(parts) > 3 else 1400.0
                config.gas.rgas = float(parts[4]) if len(parts) > 4 else 287.15
                config.gas.gas_type = GasType.VARIABLE_CP

            # 讀取時間步進類型
            line = self._read_line()
            itimst = int(line.split()[0]) if line else 3
            config.solver.time_step_type = TimeStepType(abs(itimst))

            # SSS 格式係數
            if itimst == 4 or itimst == -4:
                line = self._read_line()
                parts = line.split()
                config.solver.f1 = float(parts[0]) if len(parts) > 0 else 2.0
                config.solver.f2eff = float(parts[1]) if len(parts) > 1 else -1.0
                config.solver.f3 = float(parts[2]) if len(parts) > 2 else 0.0
                config.solver.rsmth = float(parts[3]) if len(parts) > 3 else 0.4
                config.solver.nrsmth = int(parts[4]) if len(parts) > 4 else 0

            # 人工可壓縮性參數
            if abs(itimst) >= 5:
                line = self._read_line()
                parts = line.split()
                config.solver.vsound = float(parts[0]) if len(parts) > 0 else 150.0
                config.solver.rf_ptru = float(parts[1]) if len(parts) > 1 else 0.01
                config.solver.rf_vsound = float(parts[2]) if len(parts) > 2 else 0.002
                config.solver.vs_vmax = float(parts[3]) if len(parts) > 3 else 2.0
                if abs(itimst) == 6 and len(parts) > 4:
                    config.solver.density = float(parts[4])

            # CFL 和阻尼
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.solver.cfl = float(parts[0]) if len(parts) > 0 else 0.4
            config.solver.damping = float(parts[1]) if len(parts) > 1 else 10.0
            config.solver.mach_limit = float(parts[2]) if len(parts) > 2 else 2.0

            # 重啟選項
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.solver.restart = int(parts[0]) == 1 if len(parts) > 0 else False
            config.solver.inverse_design = int(parts[1]) == 1 if len(parts) > 1 else False
            self._skip_comment_line()

            # 收斂控制
            line = self._read_line()
            parts = line.split()
            config.solver.max_steps = int(parts[0]) if len(parts) > 0 else 5000
            config.solver.convergence_limit = float(parts[1]) if len(parts) > 1 else 0.005

            # 平滑因子
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.solver.sfx_in = float(parts[0]) if len(parts) > 0 else 0.005
            config.solver.sft_in = float(parts[1]) if len(parts) > 1 else 0.005
            config.solver.fac_4th = float(parts[2]) if len(parts) > 2 else 0.8
            config.solver.nchange = (
                int(parts[3]) if len(parts) > 3 else config.solver.max_steps // 4
            )

            # 葉片排數
            self._skip_comment_line()
            config.nrows = self._read_int()

            # 網格點數
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.grid.im = int(parts[0]) if len(parts) > 0 else 37
            config.grid.km = int(parts[1]) if len(parts) > 1 else 11

            # 周向間距
            self._skip_comment_line()
            config.grid.fp = self._read_floats(config.grid.im - 1)

            # 跨向間距
            self._skip_comment_line()
            config.grid.fr = self._read_floats(config.grid.km - 1)

            # 多重網格參數
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            if len(parts) >= 6:
                config.grid.ir = int(parts[0])
                config.grid.jr = int(parts[1])
                config.grid.kr = int(parts[2])
                config.grid.irbb = int(parts[3])
                config.grid.jrbb = int(parts[4])
                config.grid.krbb = int(parts[5])

            # 多重網格時間步因子
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.grid.fblk1 = float(parts[0]) if len(parts) > 0 else 0.4
            config.grid.fblk2 = float(parts[1]) if len(parts) > 1 else 0.2
            config.grid.fblk3 = float(parts[2]) if len(parts) > 2 else 0.1

            # 混合平面參數
            self._skip_comment_line()
            line = self._read_line()
            ifmix = int(line.split()[0]) if line else 1
            config.mixing_plane.enabled = ifmix != 0

            if config.mixing_plane.enabled:
                self._skip_comment_line()
                line = self._read_line()
                parts = line.split()
                config.mixing_plane.rfmix = float(parts[0]) if len(parts) > 0 else 0.025
                config.mixing_plane.fextrap = float(parts[1]) if len(parts) > 1 else 0.8
                config.mixing_plane.fsmthb = float(parts[2]) if len(parts) > 2 else 1.0
                config.mixing_plane.fangle = float(parts[3]) if len(parts) > 3 else 0.8

            # 冷卻、放氣、粗糙度標記
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.cooling_enabled = int(parts[0]) == 1 if len(parts) > 0 else False
            config.bleed_enabled = int(parts[1]) == 1 if len(parts) > 1 else False
            config.roughness_enabled = int(parts[2]) == 1 if len(parts) > 2 else False

            # 截面數
            self._skip_comment_line()
            config.nsections = self._read_int()

            # 進口邊界條件
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            if len(parts) >= 6:
                config.inlet.use_total_pressure = int(parts[0]) == 0
                config.inlet.use_tangential_velocity = int(parts[1]) != 0
                config.inlet.use_radial_velocity = int(parts[2]) != 0
                config.inlet.use_mass_flow = int(parts[3]) != 0
                config.inlet.repeating_stage = int(parts[4]) != 0
                config.inlet.rfin = float(parts[5])

            # 出口邊界條件
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            config.exit.use_static_pressure = int(parts[0]) == 1 if len(parts) > 0 else True
            config.exit.sfexit = float(parts[1]) if len(parts) > 1 else 0.0
            config.exit.nsfexit = int(parts[2]) if len(parts) > 2 else 0
            config.exit.fp_xtrap = float(parts[3]) if len(parts) > 3 else 1.0
            config.exit.fracwave = float(parts[4]) if len(parts) > 4 else 0.0

            # 黏性模型
            self._skip_comment_line()
            line = self._read_line()
            parts = line.split()
            ilos = int(parts[0]) if len(parts) > 0 else 100
            config.viscous.model = (
                ViscousModel(ilos) if ilos in [0, 100, 150, 200] else ViscousModel.MIXING_LENGTH
            )
            config.viscous.nlos = int(parts[1]) if len(parts) > 1 else 5

            if ilos != 0:
                # 黏性參數
                self._skip_comment_line()
                line = self._read_line()
                parts = line.split()
                config.viscous.reynolds = float(parts[0]) if len(parts) > 0 else 500000.0
                config.viscous.rf_vis = float(parts[1]) if len(parts) > 1 else 0.5
                config.viscous.ftrans = float(parts[2]) if len(parts) > 2 else 0.0001
                config.viscous.turbvis_limit = float(parts[3]) if len(parts) > 3 else 3000.0

            self._file = None

        return config


class MultallOutputWriter:
    """MULTALL 輸出文件寫入器。"""

    def __init__(self, output_dir: str | Path = ".") -> None:
        """初始化寫入器。

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_results(
        self,
        config: MultallConfig,
        filename: str = "results.out",
    ) -> None:
        """寫入結果文件。

        Args:
            config: MULTALL 配置
            filename: 輸出文件名
        """
        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MULTALL - 3D 渦輪機械流場求解器\n")
            f.write(f"標題: {config.title}\n")
            f.write("=" * 80 + "\n\n")

            # 氣體性質
            f.write("氣體性質:\n")
            f.write(f"  CP = {config.gas.cp:.4f} J/(kg·K)\n")
            f.write(f"  γ  = {config.gas.gamma:.4f}\n")
            f.write(f"  R  = {config.gas.rgas:.4f} J/(kg·K)\n\n")

            # 網格信息
            f.write("網格參數:\n")
            f.write(f"  周向網格點數 (IM) = {config.grid.im}\n")
            f.write(f"  跨向網格點數 (KM) = {config.grid.km}\n")
            f.write(f"  葉片排數 (NROWS)  = {config.nrows}\n\n")

            # 求解器參數
            f.write("求解器參數:\n")
            f.write(f"  CFL 數            = {config.solver.cfl}\n")
            f.write(f"  最大時間步數      = {config.solver.max_steps}\n")
            f.write(f"  收斂準則          = {config.solver.convergence_limit}\n\n")

            # 葉片排信息
            if config.blade_rows:
                f.write("葉片排:\n")
                for i, row in enumerate(config.blade_rows, 1):
                    row_type = "轉子" if row.row_type == "R" else "定子"
                    f.write(f"  排 {i}: {row_type}, {row.n_blades} 葉片")
                    if row.row_type == "R":
                        f.write(f", {row.rpm:.0f} RPM")
                    f.write("\n")

            f.write("\n" + "=" * 80 + "\n")

    def write_log(
        self,
        step: int,
        residual: float,
        mass_flow: float,
        efficiency: float | None = None,
        filename: str = "stage.log",
    ) -> None:
        """寫入收斂日誌。

        Args:
            step: 時間步
            residual: 殘差
            mass_flow: 質量流量
            efficiency: 效率
            filename: 日誌文件名
        """
        log_file = self.output_dir / filename
        mode = "a" if log_file.exists() else "w"

        with open(log_file, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write("步數\t殘差\t\t質量流量\t效率\n")
                f.write("-" * 60 + "\n")

            eff_str = f"{efficiency:.4f}" if efficiency is not None else "N/A"
            f.write(f"{step}\t{residual:.6e}\t{mass_flow:.6f}\t{eff_str}\n")


class MultallFileHandler:
    """MULTALL 文件處理器（整合讀取和寫入）。"""

    def __init__(
        self,
        input_file: str | Path | None = None,
        output_dir: str | Path = ".",
    ) -> None:
        """初始化文件處理器。

        Args:
            input_file: 輸入文件路徑（可選）
            output_dir: 輸出目錄
        """
        self.input_file = Path(input_file) if input_file else None
        self.reader = MultallInputReader()
        self.writer = MultallOutputWriter(output_dir)

    def read_input(self) -> MultallConfig:
        """讀取輸入文件。

        Returns:
            MULTALL 配置
        """
        if self.input_file is None:
            msg = "未指定輸入文件"
            raise ValueError(msg)
        return self.reader.read(self.input_file)

    def write_outputs(self, config: MultallConfig) -> None:
        """寫入所有輸出文件。

        Args:
            config: MULTALL 配置
        """
        self.writer.write_results(config)
