# -*- coding: utf-8 -*-
"""批量處理工具。

提供參數掃描和批量計算功能。
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ParameterRange:
    """參數範圍定義。

    Attributes:
        name: 參數名稱
        values: 參數值列表或範圍
        path: 參數在配置中的路徑（用點分隔）
    """

    name: str
    values: list[float] | NDArray[np.floating]
    path: str = ""

    @classmethod
    def from_range(
        cls,
        name: str,
        start: float,
        stop: float,
        num: int = 10,
        path: str = "",
    ) -> ParameterRange:
        """從範圍創建參數。

        Args:
            name: 參數名稱
            start: 起始值
            stop: 結束值
            num: 點數
            path: 配置路徑

        Returns:
            ParameterRange 實例
        """
        return cls(
            name=name,
            values=np.linspace(start, stop, num),
            path=path or name,
        )

    @classmethod
    def from_list(
        cls,
        name: str,
        values: list[float],
        path: str = "",
    ) -> ParameterRange:
        """從列表創建參數。

        Args:
            name: 參數名稱
            values: 值列表
            path: 配置路徑

        Returns:
            ParameterRange 實例
        """
        return cls(
            name=name,
            values=values,
            path=path or name,
        )


@dataclass
class BatchResult:
    """批量計算結果。

    Attributes:
        parameters: 參數組合
        results: 計算結果列表
        success_count: 成功計數
        failure_count: 失敗計數
        errors: 錯誤信息
    """

    parameters: list[dict[str, float]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dataframe_dict(self) -> dict[str, list]:
        """轉換為 DataFrame 字典格式。

        Returns:
            可用於創建 DataFrame 的字典
        """
        if not self.parameters:
            return {}

        # 收集所有參數鍵
        param_keys = list(self.parameters[0].keys())

        # 收集所有結果鍵
        result_keys = []
        for r in self.results:
            if r:
                result_keys = list(r.keys())
                break

        # 構建字典
        data: dict[str, list] = {k: [] for k in param_keys + result_keys}

        for params, result in zip(self.parameters, self.results):
            for k in param_keys:
                data[k].append(params.get(k))
            for k in result_keys:
                data[k].append(result.get(k) if result else None)

        return data


class BatchProcessor:
    """批量處理器。

    執行參數掃描和批量計算。
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        compute_func: Callable[[dict[str, Any]], dict[str, Any]],
        max_workers: int | None = None,
    ) -> None:
        """初始化批量處理器。

        Args:
            base_config: 基礎配置
            compute_func: 計算函數
            max_workers: 最大工作進程數
        """
        self.base_config = base_config.copy()
        self.compute_func = compute_func
        self.max_workers = max_workers

    def run_sweep(
        self,
        parameters: list[ParameterRange],
        mode: str = "grid",
        callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """執行參數掃描。

        Args:
            parameters: 參數範圍列表
            mode: 掃描模式 ("grid" 或 "zip")
            callback: 進度回調函數

        Returns:
            批量計算結果
        """
        # 生成參數組合
        if mode == "grid":
            combinations = self._generate_grid_combinations(parameters)
        else:
            combinations = self._generate_zip_combinations(parameters)

        return self._run_batch(combinations, callback)

    def run_batch(
        self,
        configs: list[dict[str, Any]],
        callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """執行批量計算。

        Args:
            configs: 配置列表
            callback: 進度回調函數

        Returns:
            批量計算結果
        """
        combinations = []
        for config in configs:
            # 合併基礎配置
            merged = self._deep_merge(self.base_config.copy(), config)
            combinations.append((config, merged))

        return self._run_batch(combinations, callback)

    def _generate_grid_combinations(
        self,
        parameters: list[ParameterRange],
    ) -> list[tuple[dict[str, float], dict[str, Any]]]:
        """生成網格組合。"""
        # 獲取所有參數的值
        param_values = [list(p.values) for p in parameters]
        param_names = [p.name for p in parameters]
        param_paths = [p.path for p in parameters]

        combinations = []
        for values in itertools.product(*param_values):
            # 參數字典
            params = dict(zip(param_names, values))

            # 更新配置
            config = self.base_config.copy()
            for path, value in zip(param_paths, values):
                self._set_nested(config, path, value)

            combinations.append((params, config))

        return combinations

    def _generate_zip_combinations(
        self,
        parameters: list[ParameterRange],
    ) -> list[tuple[dict[str, float], dict[str, Any]]]:
        """生成 zip 組合（一一對應）。"""
        # 確保所有參數長度相同
        lengths = [len(p.values) for p in parameters]
        if len(set(lengths)) > 1:
            raise ValueError("所有參數必須有相同數量的值")

        param_names = [p.name for p in parameters]
        param_paths = [p.path for p in parameters]

        combinations = []
        for values in zip(*[p.values for p in parameters]):
            params = dict(zip(param_names, values))

            config = self.base_config.copy()
            for path, value in zip(param_paths, values):
                self._set_nested(config, path, value)

            combinations.append((params, config))

        return combinations

    def _run_batch(
        self,
        combinations: list[tuple[dict[str, float], dict[str, Any]]],
        callback: Callable[[int, int], None] | None,
    ) -> BatchResult:
        """執行批量計算。"""
        result = BatchResult()
        total = len(combinations)

        # 單線程執行（更穩定）
        for i, (params, config) in enumerate(combinations):
            result.parameters.append(params)

            try:
                output = self.compute_func(config)
                result.results.append(output)
                result.success_count += 1
            except Exception as e:
                result.results.append({})
                result.failure_count += 1
                result.errors.append(f"參數 {params}: {e}")

            if callback:
                callback(i + 1, total)

        return result

    def _set_nested(self, d: dict, path: str, value: Any) -> None:
        """設定嵌套字典的值。"""
        keys = path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """深度合併字典。"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


def parameter_sweep(
    base_config: dict[str, Any],
    parameters: list[ParameterRange],
    compute_func: Callable[[dict[str, Any]], dict[str, Any]],
    mode: str = "grid",
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """參數掃描的便捷函數。

    Args:
        base_config: 基礎配置
        parameters: 參數範圍列表
        compute_func: 計算函數
        mode: 掃描模式
        progress_callback: 進度回調

    Returns:
        批量計算結果

    Example:
        >>> params = [
        ...     ParameterRange.from_range("phi", 0.4, 0.8, 5, "stages.0.phi"),
        ...     ParameterRange.from_range("psi", 1.5, 2.5, 5, "stages.0.psi"),
        ... ]
        >>> result = parameter_sweep(config, params, compute_meangen)
    """
    processor = BatchProcessor(base_config, compute_func)
    return processor.run_sweep(parameters, mode, progress_callback)


def save_batch_results(
    result: BatchResult,
    output_dir: str | Path,
    prefix: str = "batch",
) -> dict[str, Path]:
    """儲存批量計算結果。

    Args:
        result: 批量計算結果
        output_dir: 輸出目錄
        prefix: 文件前綴

    Returns:
        輸出文件路徑字典
    """
    from multall_turbomachinery_design.utils.export import DataExporter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = DataExporter(output_dir)
    paths = {}

    # 導出為 CSV
    df_dict = result.to_dataframe_dict()
    if df_dict:
        paths["csv"] = exporter.to_csv(df_dict, f"{prefix}_results.csv")

    # 導出為 JSON
    summary = {
        "total_runs": result.success_count + result.failure_count,
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "parameters": result.parameters,
        "results": result.results,
        "errors": result.errors,
    }
    paths["json"] = exporter.to_json(summary, f"{prefix}_summary.json")

    return paths
