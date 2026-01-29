# -*- coding: utf-8 -*-
"""數據導出工具。

提供計算結果的導出功能，支援多種格式。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DataExporter:
    """數據導出器。

    支援導出為 CSV、JSON、NPZ 等格式。
    """

    def __init__(self, output_dir: str | Path = ".") -> None:
        """初始化導出器。

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_csv(
        self,
        data: dict[str, list | NDArray] | list[dict],
        filename: str,
        headers: list[str] | None = None,
    ) -> Path:
        """導出為 CSV 格式。

        Args:
            data: 數據（字典或列表）
            filename: 文件名
            headers: 標頭（可選）

        Returns:
            輸出文件路徑
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".csv")

        if isinstance(data, dict):
            # 字典格式：每個鍵是一列
            if headers is None:
                headers = list(data.keys())

            # 轉換為行格式
            n_rows = len(next(iter(data.values())))
            rows = []
            for i in range(n_rows):
                row = [data[h][i] if h in data else "" for h in headers]
                rows.append(row)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        else:
            # 列表格式：每個元素是一行
            if headers is None and data:
                headers = list(data[0].keys())

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)

        return filepath

    def to_json(
        self,
        data: Any,
        filename: str,
        indent: int = 2,
    ) -> Path:
        """導出為 JSON 格式。

        Args:
            data: 數據
            filename: 文件名
            indent: 縮進空格數

        Returns:
            輸出文件路徑
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".json")

        # 轉換數據
        serializable = self._make_serializable(data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=indent, ensure_ascii=False)

        return filepath

    def to_npz(
        self,
        data: dict[str, NDArray],
        filename: str,
        compressed: bool = True,
    ) -> Path:
        """導出為 NPZ 格式（NumPy 壓縮檔案）。

        Args:
            data: 數據字典
            filename: 文件名
            compressed: 是否壓縮

        Returns:
            輸出文件路徑
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".npz")

        if compressed:
            np.savez_compressed(filepath, **data)
        else:
            np.savez(filepath, **data)

        return filepath

    def to_tecplot(
        self,
        x: NDArray,
        y: NDArray,
        variables: dict[str, NDArray],
        filename: str,
        title: str = "MULTALL Data",
        zone_name: str = "Zone 1",
    ) -> Path:
        """導出為 Tecplot ASCII 格式。

        Args:
            x: X 座標
            y: Y 座標
            variables: 變量字典
            filename: 文件名
            title: 標題
            zone_name: 區域名稱

        Returns:
            輸出文件路徑
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".dat")

        # 確保數據是 2D
        if x.ndim == 1:
            ni = len(x)
            nj = 1
        else:
            ni, nj = x.shape

        var_names = ["X", "Y"] + list(variables.keys())

        with open(filepath, "w", encoding="utf-8") as f:
            # 標題
            f.write(f'TITLE = "{title}"\n')

            # 變量名
            var_str = ", ".join(f'"{v}"' for v in var_names)
            f.write(f"VARIABLES = {var_str}\n")

            # 區域
            f.write(f'ZONE T="{zone_name}", I={ni}, J={nj}, F=POINT\n')

            # 數據
            x_flat = x.flatten()
            y_flat = y.flatten()
            var_flat = {k: v.flatten() for k, v in variables.items()}

            for i in range(len(x_flat)):
                values = [x_flat[i], y_flat[i]]
                values.extend(var_flat[k][i] for k in variables)
                line = " ".join(f"{v:15.8e}" for v in values)
                f.write(line + "\n")

        return filepath

    def to_vtk(
        self,
        x: NDArray,
        y: NDArray,
        z: NDArray,
        scalars: dict[str, NDArray] | None = None,
        vectors: dict[str, tuple[NDArray, NDArray, NDArray]] | None = None,
        filename: str = "output",
    ) -> Path:
        """導出為 VTK 結構化網格格式。

        Args:
            x: X 座標 (3D array)
            y: Y 座標 (3D array)
            z: Z 座標 (3D array)
            scalars: 純量場字典
            vectors: 向量場字典
            filename: 文件名

        Returns:
            輸出文件路徑
        """
        filepath = self.output_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix(".vtk")

        ni, nj, nk = x.shape
        n_points = ni * nj * nk

        with open(filepath, "w", encoding="utf-8") as f:
            # VTK 標頭
            f.write("# vtk DataFile Version 3.0\n")
            f.write("MULTALL Output\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {ni} {nj} {nk}\n")
            f.write(f"POINTS {n_points} float\n")

            # 點座標
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        f.write(f"{x[i, j, k]} {y[i, j, k]} {z[i, j, k]}\n")

            # 點數據
            if scalars or vectors:
                f.write(f"\nPOINT_DATA {n_points}\n")

            # 純量場
            if scalars:
                for name, data in scalars.items():
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for k in range(nk):
                        for j in range(nj):
                            for i in range(ni):
                                f.write(f"{data[i, j, k]}\n")

            # 向量場
            if vectors:
                for name, (vx, vy, vz) in vectors.items():
                    f.write(f"VECTORS {name} float\n")
                    for k in range(nk):
                        for j in range(nj):
                            for i in range(ni):
                                f.write(f"{vx[i, j, k]} {vy[i, j, k]} {vz[i, j, k]}\n")

        return filepath

    def _make_serializable(self, obj: Any) -> Any:
        """將物件轉換為 JSON 可序列化格式。"""
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        if is_dataclass(obj):
            return self._make_serializable(asdict(obj))
        # 其他類型嘗試轉換為字串
        return str(obj)


def export_to_csv(
    data: dict[str, list | NDArray] | list[dict],
    filepath: str | Path,
    headers: list[str] | None = None,
) -> Path:
    """導出為 CSV 的便捷函數。

    Args:
        data: 數據
        filepath: 文件路徑
        headers: 標頭

    Returns:
        輸出文件路徑
    """
    filepath = Path(filepath)
    exporter = DataExporter(filepath.parent)
    return exporter.to_csv(data, filepath.name, headers)


def export_to_json(
    data: Any,
    filepath: str | Path,
    indent: int = 2,
) -> Path:
    """導出為 JSON 的便捷函數。

    Args:
        data: 數據
        filepath: 文件路徑
        indent: 縮進

    Returns:
        輸出文件路徑
    """
    filepath = Path(filepath)
    exporter = DataExporter(filepath.parent)
    return exporter.to_json(data, filepath.name, indent)


def export_performance_report(
    performance: dict[str, float],
    filepath: str | Path,
    title: str = "性能報告",
) -> Path:
    """導出性能報告。

    Args:
        performance: 性能數據
        filepath: 文件路徑
        title: 報告標題

    Returns:
        輸出文件路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n\n")

        for key, value in performance.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 50 + "\n")

    return filepath
