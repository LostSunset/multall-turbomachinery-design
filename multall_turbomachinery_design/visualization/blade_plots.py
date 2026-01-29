# -*- coding: utf-8 -*-
"""葉片視覺化。

提供葉片截面和 3D 表面的繪圖功能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class BladePlotter:
    """葉片繪圖器。

    繪製 2D 葉片截面和 3D 葉片表面。
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """初始化繪圖器。

        Args:
            figsize: 圖形尺寸
            dpi: 解析度
        """
        self.figsize = figsize
        self.dpi = dpi
        self._colors = {
            "upper": "#1976D2",  # 藍色 - 吸力面
            "lower": "#D32F2F",  # 紅色 - 壓力面
            "camber": "#388E3C",  # 綠色 - 中弧線
            "chord": "#757575",  # 灰色 - 弦線
        }

    def plot_profile(
        self,
        x_upper: NDArray[np.floating],
        y_upper: NDArray[np.floating],
        x_lower: NDArray[np.floating],
        y_lower: NDArray[np.floating],
        x_camber: NDArray[np.floating] | None = None,
        y_camber: NDArray[np.floating] | None = None,
        title: str = "葉片截面",
        show_chord: bool = True,
        show_angles: bool = True,
    ) -> Figure:
        """繪製 2D 葉片截面。

        Args:
            x_upper: 吸力面 x 座標
            y_upper: 吸力面 y 座標
            x_lower: 壓力面 x 座標
            y_lower: 壓力面 y 座標
            x_camber: 中弧線 x 座標（可選）
            y_camber: 中弧線 y 座標（可選）
            title: 圖標題
            show_chord: 是否顯示弦線
            show_angles: 是否顯示角度

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 繪製葉片輪廓
        ax.plot(x_upper, y_upper, color=self._colors["upper"], lw=2, label="吸力面")
        ax.plot(x_lower, y_lower, color=self._colors["lower"], lw=2, label="壓力面")

        # 填充葉片區域
        x_profile = np.concatenate([x_upper, x_lower[::-1]])
        y_profile = np.concatenate([y_upper, y_lower[::-1]])
        ax.fill(x_profile, y_profile, alpha=0.2, color="#90CAF9")

        # 繪製中弧線
        if x_camber is not None and y_camber is not None:
            ax.plot(
                x_camber,
                y_camber,
                "--",
                color=self._colors["camber"],
                lw=1.5,
                label="中弧線",
            )

        # 繪製弦線
        if show_chord:
            x_le = x_upper[0]
            y_le = (y_upper[0] + y_lower[0]) / 2
            x_te = x_upper[-1]
            y_te = (y_upper[-1] + y_lower[-1]) / 2
            ax.plot(
                [x_le, x_te],
                [y_le, y_te],
                "-.",
                color=self._colors["chord"],
                lw=1,
                label="弦線",
            )

            # 計算弦長
            chord = np.sqrt((x_te - x_le) ** 2 + (y_te - y_le) ** 2)
            ax.text(
                (x_le + x_te) / 2,
                (y_le + y_te) / 2 - 0.02,
                f"弦長: {chord:.4f}",
                fontsize=9,
                ha="center",
            )

        # 繪製角度
        if show_angles and x_camber is not None and y_camber is not None:
            self._add_angle_annotations(ax, x_camber, y_camber)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        return fig

    def _add_angle_annotations(
        self,
        ax: Axes,
        x_camber: NDArray[np.floating],
        y_camber: NDArray[np.floating],
    ) -> None:
        """添加角度標註。

        Args:
            ax: matplotlib Axes
            x_camber: 中弧線 x 座標
            y_camber: 中弧線 y 座標
        """
        # 入口角（前幾個點的切線）
        dx_in = x_camber[1] - x_camber[0]
        dy_in = y_camber[1] - y_camber[0]
        inlet_angle = np.degrees(np.arctan2(dy_in, dx_in))

        # 出口角（後幾個點的切線）
        dx_out = x_camber[-1] - x_camber[-2]
        dy_out = y_camber[-1] - y_camber[-2]
        outlet_angle = np.degrees(np.arctan2(dy_out, dx_out))

        # 標註入口角
        ax.annotate(
            f"β₁={inlet_angle:.1f}°",
            xy=(x_camber[0], y_camber[0]),
            xytext=(x_camber[0] - 0.02, y_camber[0] + 0.03),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

        # 標註出口角
        ax.annotate(
            f"β₂={outlet_angle:.1f}°",
            xy=(x_camber[-1], y_camber[-1]),
            xytext=(x_camber[-1] + 0.02, y_camber[-1] + 0.03),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )

    def plot_cascade(
        self,
        x_upper: NDArray[np.floating],
        y_upper: NDArray[np.floating],
        x_lower: NDArray[np.floating],
        y_lower: NDArray[np.floating],
        pitch: float,
        n_blades: int = 3,
        title: str = "葉柵",
    ) -> Figure:
        """繪製葉柵（多個葉片）。

        Args:
            x_upper: 吸力面 x 座標
            y_upper: 吸力面 y 座標
            x_lower: 壓力面 x 座標
            y_lower: 壓力面 y 座標
            pitch: 節距
            n_blades: 葉片數量
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 繪製多個葉片
        for i in range(n_blades):
            offset = i * pitch
            ax.plot(x_upper, y_upper + offset, color=self._colors["upper"], lw=2)
            ax.plot(x_lower, y_lower + offset, color=self._colors["lower"], lw=2)

            # 填充
            x_profile = np.concatenate([x_upper, x_lower[::-1]])
            y_profile = np.concatenate([y_upper + offset, y_lower[::-1] + offset])
            ax.fill(x_profile, y_profile, alpha=0.3, color="#90CAF9")

        # 添加節距標註
        mid_idx = len(x_upper) // 2
        ax.annotate(
            "",
            xy=(x_upper[mid_idx], y_upper[mid_idx] + pitch),
            xytext=(x_upper[mid_idx], y_upper[mid_idx]),
            arrowprops=dict(arrowstyle="<->", color="green", lw=1.5),
        )
        ax.text(
            x_upper[mid_idx] + 0.01,
            y_upper[mid_idx] + pitch / 2,
            f"s={pitch:.4f}",
            fontsize=9,
            color="green",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_3d_surface(
        self,
        x: NDArray[np.floating],
        r: NDArray[np.floating],
        theta: NDArray[np.floating],
        title: str = "葉片 3D 表面",
    ) -> Figure:
        """繪製 3D 葉片表面。

        Args:
            x: 軸向座標 (2D array)
            r: 半徑座標 (2D array)
            theta: 周向角度座標 (2D array)
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # 轉換為笛卡爾座標
        y = r * np.cos(theta)
        z = r * np.sin(theta)

        # 繪製表面
        ax.plot_surface(x, y, z, cmap="coolwarm", alpha=0.8, edgecolor="none")

        ax.set_xlabel("X (軸向)")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        plt.tight_layout()
        return fig


def plot_blade_profile(
    x_upper: NDArray[np.floating],
    y_upper: NDArray[np.floating],
    x_lower: NDArray[np.floating],
    y_lower: NDArray[np.floating],
    x_camber: NDArray[np.floating] | None = None,
    y_camber: NDArray[np.floating] | None = None,
    title: str = "葉片截面",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製葉片截面的便捷函數。

    Args:
        x_upper: 吸力面 x 座標
        y_upper: 吸力面 y 座標
        x_lower: 壓力面 x 座標
        y_lower: 壓力面 y 座標
        x_camber: 中弧線 x 座標（可選）
        y_camber: 中弧線 y 座標（可選）
        title: 圖標題
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = BladePlotter()
    fig = plotter.plot_profile(x_upper, y_upper, x_lower, y_lower, x_camber, y_camber, title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_blade_surface(
    x: NDArray[np.floating],
    r: NDArray[np.floating],
    theta: NDArray[np.floating],
    title: str = "葉片 3D 表面",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製 3D 葉片表面的便捷函數。

    Args:
        x: 軸向座標 (2D array)
        r: 半徑座標 (2D array)
        theta: 周向角度座標 (2D array)
        title: 圖標題
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = BladePlotter()
    fig = plotter.plot_3d_surface(x, r, theta, title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
