# -*- coding: utf-8 -*-
"""收斂歷史視覺化。

提供收斂歷史和殘差的繪圖功能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class ConvergencePlotter:
    """收斂歷史繪圖器。

    繪製殘差歷史、性能收斂等圖表。
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (12, 8),
        dpi: int = 100,
    ) -> None:
        """初始化繪圖器。

        Args:
            figsize: 圖形尺寸
            dpi: 解析度
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_residuals(
        self,
        iterations: NDArray[np.integer] | list[int],
        residuals: dict[str, NDArray[np.floating] | list[float]],
        title: str = "殘差收斂歷史",
        log_scale: bool = True,
        target_residual: float | None = None,
    ) -> Figure:
        """繪製殘差收斂歷史。

        Args:
            iterations: 迭代次數
            residuals: 殘差字典 {名稱: 數據}
            title: 圖標題
            log_scale: 是否使用對數刻度
            target_residual: 目標殘差線（可選）

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 繪製各殘差
        for name, data in residuals.items():
            ax.plot(iterations, data, lw=1.5, label=name)

        # 目標殘差線
        if target_residual is not None:
            ax.axhline(
                y=target_residual,
                color="r",
                linestyle="--",
                lw=1,
                label=f"目標: {target_residual:.0e}",
            )

        ax.set_xlabel("迭代次數")
        ax.set_ylabel("殘差")
        ax.set_title(title)

        if log_scale:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        return fig

    def plot_performance(
        self,
        iterations: NDArray[np.integer] | list[int],
        performance: dict[str, NDArray[np.floating] | list[float]],
        title: str = "性能收斂歷史",
    ) -> Figure:
        """繪製性能收斂歷史。

        Args:
            iterations: 迭代次數
            performance: 性能數據字典 {名稱: 數據}
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        n_metrics = len(performance)
        fig, axes = plt.subplots(
            n_metrics, 1, figsize=(self.figsize[0], self.figsize[1] * n_metrics / 3), dpi=self.dpi
        )

        if n_metrics == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, performance.items()):
            ax.plot(iterations, data, "b-", lw=1.5)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

            # 顯示最終值
            final_value = data[-1] if hasattr(data, "__getitem__") else data
            ax.axhline(y=final_value, color="r", linestyle="--", lw=0.5, alpha=0.5)
            ax.text(
                iterations[-1],
                final_value,
                f" {final_value:.4f}",
                va="center",
                fontsize=9,
            )

        axes[-1].set_xlabel("迭代次數")
        axes[0].set_title(title)

        plt.tight_layout()
        return fig

    def plot_combined(
        self,
        iterations: NDArray[np.integer] | list[int],
        residual: NDArray[np.floating] | list[float],
        efficiency: NDArray[np.floating] | list[float],
        mass_flow: NDArray[np.floating] | list[float],
        title: str = "收斂歷史",
    ) -> Figure:
        """繪製組合收斂圖（殘差 + 效率 + 質量流率）。

        Args:
            iterations: 迭代次數
            residual: 殘差數據
            efficiency: 效率數據
            mass_flow: 質量流率數據
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)

        # 殘差（對數刻度）
        axes[0].semilogy(iterations, residual, "b-", lw=1.5)
        axes[0].set_ylabel("殘差")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title)

        # 效率
        axes[1].plot(iterations, efficiency, "g-", lw=1.5)
        axes[1].set_ylabel("等熵效率")
        axes[1].grid(True, alpha=0.3)
        final_eff = efficiency[-1] if hasattr(efficiency, "__getitem__") else efficiency
        axes[1].axhline(y=final_eff, color="r", linestyle="--", lw=0.5, alpha=0.5)

        # 質量流率
        axes[2].plot(iterations, mass_flow, "r-", lw=1.5)
        axes[2].set_ylabel("質量流率 (kg/s)")
        axes[2].set_xlabel("迭代次數")
        axes[2].grid(True, alpha=0.3)
        final_mf = mass_flow[-1] if hasattr(mass_flow, "__getitem__") else mass_flow
        axes[2].axhline(y=final_mf, color="b", linestyle="--", lw=0.5, alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_stage_performance(
        self,
        stages: list[int],
        efficiency: list[float],
        pressure_ratio: list[float],
        temperature_ratio: list[float],
        title: str = "各級性能",
    ) -> Figure:
        """繪製各級性能對比。

        Args:
            stages: 級數列表
            efficiency: 各級效率
            pressure_ratio: 各級壓比
            temperature_ratio: 各級溫比
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)

        x = np.arange(len(stages))
        width = 0.6

        # 效率
        axes[0].bar(x, efficiency, width, color="#2196F3")
        axes[0].set_xlabel("級")
        axes[0].set_ylabel("等熵效率")
        axes[0].set_title("效率")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stages)
        axes[0].grid(True, alpha=0.3, axis="y")

        # 壓比
        axes[1].bar(x, pressure_ratio, width, color="#4CAF50")
        axes[1].set_xlabel("級")
        axes[1].set_ylabel("總壓比")
        axes[1].set_title("壓比")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(stages)
        axes[1].grid(True, alpha=0.3, axis="y")

        # 溫比
        axes[2].bar(x, temperature_ratio, width, color="#FF9800")
        axes[2].set_xlabel("級")
        axes[2].set_ylabel("總溫比")
        axes[2].set_title("溫比")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(stages)
        axes[2].grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig


def plot_convergence_history(
    iterations: NDArray[np.integer] | list[int],
    residual: NDArray[np.floating] | list[float],
    efficiency: NDArray[np.floating] | list[float] | None = None,
    mass_flow: NDArray[np.floating] | list[float] | None = None,
    title: str = "收斂歷史",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製收斂歷史的便捷函數。

    Args:
        iterations: 迭代次數
        residual: 殘差數據
        efficiency: 效率數據（可選）
        mass_flow: 質量流率數據（可選）
        title: 圖標題
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = ConvergencePlotter()

    if efficiency is not None and mass_flow is not None:
        fig = plotter.plot_combined(iterations, residual, efficiency, mass_flow, title)
    else:
        fig = plotter.plot_residuals(iterations, {"殘差": residual}, title, log_scale=True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_residual_history(
    iterations: NDArray[np.integer] | list[int],
    residuals: dict[str, NDArray[np.floating] | list[float]],
    title: str = "殘差收斂歷史",
    target_residual: float | None = 1e-6,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製殘差歷史的便捷函數。

    Args:
        iterations: 迭代次數
        residuals: 殘差字典 {名稱: 數據}
        title: 圖標題
        target_residual: 目標殘差線
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = ConvergencePlotter()
    fig = plotter.plot_residuals(
        iterations, residuals, title, log_scale=True, target_residual=target_residual
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
