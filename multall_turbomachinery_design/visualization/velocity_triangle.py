# -*- coding: utf-8 -*-
"""速度三角形視覺化。

提供速度三角形的繪圖功能。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class VelocityComponents:
    """速度分量數據。

    Attributes:
        vx: 軸向速度 (m/s)
        vt: 切向速度 (m/s)
        u: 葉片速度 (m/s)
    """

    vx: float
    vt: float
    u: float

    @property
    def v(self) -> float:
        """絕對速度大小。"""
        return np.sqrt(self.vx**2 + self.vt**2)

    @property
    def w(self) -> float:
        """相對速度大小。"""
        wt = self.vt - self.u
        return np.sqrt(self.vx**2 + wt**2)

    @property
    def alpha(self) -> float:
        """絕對流動角（度）。"""
        return np.degrees(np.arctan2(self.vt, self.vx))

    @property
    def beta(self) -> float:
        """相對流動角（度）。"""
        wt = self.vt - self.u
        return np.degrees(np.arctan2(wt, self.vx))


class VelocityTrianglePlotter:
    """速度三角形繪圖器。

    繪製入口和出口的速度三角形。
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (12, 5),
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
            "v": "#2196F3",  # 藍色 - 絕對速度
            "w": "#F44336",  # 紅色 - 相對速度
            "u": "#4CAF50",  # 綠色 - 葉片速度
            "vx": "#9C27B0",  # 紫色 - 軸向速度
        }

    def plot(
        self,
        inlet: VelocityComponents,
        outlet: VelocityComponents,
        title: str = "速度三角形",
    ) -> Figure:
        """繪製入口和出口速度三角形。

        Args:
            inlet: 入口速度分量
            outlet: 出口速度分量
            title: 圖標題

        Returns:
            matplotlib Figure 物件
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        self._plot_triangle(axes[0], inlet, "入口")
        self._plot_triangle(axes[1], outlet, "出口")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig

    def _plot_triangle(
        self,
        ax: Axes,
        vel: VelocityComponents,
        label: str,
    ) -> None:
        """繪製單個速度三角形。

        Args:
            ax: matplotlib Axes
            vel: 速度分量
            label: 標籤（入口/出口）
        """
        # 原點
        origin = np.array([0, 0])

        # 絕對速度向量 V
        v_vec = np.array([vel.vx, vel.vt])

        # 葉片速度向量 U（純切向）
        u_vec = np.array([0, vel.u])

        # 繪製速度向量
        # V: 從原點到 (vx, vt)
        ax.annotate(
            "",
            xy=v_vec,
            xytext=origin,
            arrowprops=dict(arrowstyle="->", color=self._colors["v"], lw=2),
        )
        ax.text(
            v_vec[0] / 2 + 5,
            v_vec[1] / 2,
            f"V={vel.v:.1f}",
            color=self._colors["v"],
            fontsize=10,
        )

        # U: 從原點到 (0, u)
        ax.annotate(
            "",
            xy=u_vec,
            xytext=origin,
            arrowprops=dict(arrowstyle="->", color=self._colors["u"], lw=2),
        )
        ax.text(
            -15,
            vel.u / 2,
            f"U={vel.u:.1f}",
            color=self._colors["u"],
            fontsize=10,
        )

        # W: 從 U 到 V（相對速度）
        ax.annotate(
            "",
            xy=v_vec,
            xytext=u_vec,
            arrowprops=dict(arrowstyle="->", color=self._colors["w"], lw=2),
        )
        ax.text(
            (u_vec[0] + v_vec[0]) / 2 + 5,
            (u_vec[1] + v_vec[1]) / 2,
            f"W={vel.w:.1f}",
            color=self._colors["w"],
            fontsize=10,
        )

        # 繪製軸向速度（虛線）
        ax.plot([0, vel.vx], [0, 0], "--", color=self._colors["vx"], lw=1)
        ax.plot([vel.vx, vel.vx], [0, vel.vt], "--", color=self._colors["vx"], lw=1)

        # 添加角度標註
        self._add_angle_arc(ax, origin, vel.alpha, 30, self._colors["v"], f"α={vel.alpha:.1f}°")
        self._add_angle_arc(ax, u_vec, vel.beta, 25, self._colors["w"], f"β={vel.beta:.1f}°")

        # 設定軸
        ax.set_xlabel("軸向速度 Vx (m/s)")
        ax.set_ylabel("切向速度 Vt (m/s)")
        ax.set_title(f"{label} 速度三角形")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        # 添加圖例
        ax.plot([], [], color=self._colors["v"], lw=2, label="絕對速度 V")
        ax.plot([], [], color=self._colors["w"], lw=2, label="相對速度 W")
        ax.plot([], [], color=self._colors["u"], lw=2, label="葉片速度 U")
        ax.legend(loc="best", fontsize=8)

    def _add_angle_arc(
        self,
        ax: Axes,
        center: np.ndarray,
        angle: float,
        radius: float,
        color: str,
        label: str,
    ) -> None:
        """添加角度弧線標註。

        Args:
            ax: matplotlib Axes
            center: 弧線中心
            angle: 角度（度）
            radius: 弧線半徑
            color: 顏色
            label: 標籤
        """
        if abs(angle) < 1:
            return

        # 創建弧線
        theta1 = 0 if angle > 0 else angle
        theta2 = angle if angle > 0 else 0
        theta = np.linspace(np.radians(theta1), np.radians(theta2), 30)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ax.plot(x, y, color=color, lw=1)

        # 添加標籤
        mid_angle = np.radians((theta1 + theta2) / 2)
        label_x = center[0] + (radius + 10) * np.cos(mid_angle)
        label_y = center[1] + (radius + 10) * np.sin(mid_angle)
        ax.text(label_x, label_y, label, color=color, fontsize=8)


def plot_velocity_triangle(
    inlet: VelocityComponents | dict,
    outlet: VelocityComponents | dict,
    title: str = "速度三角形",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製速度三角形的便捷函數。

    Args:
        inlet: 入口速度分量（VelocityComponents 或 dict）
        outlet: 出口速度分量（VelocityComponents 或 dict）
        title: 圖標題
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    # 轉換字典為 VelocityComponents
    if isinstance(inlet, dict):
        inlet = VelocityComponents(**inlet)
    if isinstance(outlet, dict):
        outlet = VelocityComponents(**outlet)

    plotter = VelocityTrianglePlotter()
    fig = plotter.plot(inlet, outlet, title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
