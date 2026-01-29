# -*- coding: utf-8 -*-
"""流場視覺化。

提供流場等值線和流線的繪圖功能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class FlowPlotter:
    """流場繪圖器。

    繪製流場等值線、流線和向量場。
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

    def plot_contour(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        data: NDArray[np.floating],
        title: str = "流場等值線",
        cmap: str = "jet",
        levels: int = 20,
        colorbar_label: str = "",
        blade_x: NDArray[np.floating] | None = None,
        blade_y: NDArray[np.floating] | None = None,
    ) -> Figure:
        """繪製等值線圖。

        Args:
            x: x 座標 (2D array)
            y: y 座標 (2D array)
            data: 數據場 (2D array)
            title: 圖標題
            cmap: 色彩映射
            levels: 等值線數量
            colorbar_label: 色標標籤
            blade_x: 葉片 x 座標（可選）
            blade_y: 葉片 y 座標（可選）

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 繪製等值線
        cf = ax.contourf(x, y, data, levels=levels, cmap=cmap)
        ax.contour(x, y, data, levels=levels, colors="k", linewidths=0.3, alpha=0.5)

        # 添加色標
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
        if colorbar_label:
            cbar.set_label(colorbar_label)

        # 繪製葉片
        if blade_x is not None and blade_y is not None:
            ax.fill(blade_x, blade_y, color="gray", alpha=0.8)
            ax.plot(blade_x, blade_y, "k-", lw=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.set_aspect("equal")

        plt.tight_layout()
        return fig

    def plot_streamlines(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        title: str = "流線圖",
        density: float = 1.5,
        color_by: NDArray[np.floating] | None = None,
        cmap: str = "viridis",
        blade_x: NDArray[np.floating] | None = None,
        blade_y: NDArray[np.floating] | None = None,
    ) -> Figure:
        """繪製流線圖。

        Args:
            x: x 座標 (1D or 2D array)
            y: y 座標 (1D or 2D array)
            u: x 方向速度 (2D array)
            v: y 方向速度 (2D array)
            title: 圖標題
            density: 流線密度
            color_by: 用於著色的數據（可選）
            cmap: 色彩映射
            blade_x: 葉片 x 座標（可選）
            blade_y: 葉片 y 座標（可選）

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 確保座標是 1D
        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]

        # 計算速度大小
        speed = np.sqrt(u**2 + v**2)

        # 繪製流線
        if color_by is not None:
            strm = ax.streamplot(
                x,
                y,
                u,
                v,
                density=density,
                color=color_by,
                cmap=cmap,
                linewidth=1,
            )
            fig.colorbar(strm.lines, ax=ax, shrink=0.8)
        else:
            ax.streamplot(
                x,
                y,
                u,
                v,
                density=density,
                color=speed,
                cmap=cmap,
                linewidth=1,
            )

        # 繪製葉片
        if blade_x is not None and blade_y is not None:
            ax.fill(blade_x, blade_y, color="gray", alpha=0.8)
            ax.plot(blade_x, blade_y, "k-", lw=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.set_aspect("equal")

        plt.tight_layout()
        return fig

    def plot_vectors(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        title: str = "向量場",
        skip: int = 1,
        scale: float | None = None,
        color_by: NDArray[np.floating] | None = None,
        cmap: str = "viridis",
    ) -> Figure:
        """繪製向量場圖。

        Args:
            x: x 座標 (2D array)
            y: y 座標 (2D array)
            u: x 方向速度 (2D array)
            v: y 方向速度 (2D array)
            title: 圖標題
            skip: 跳過的點數（用於稀疏化）
            scale: 向量縮放比例
            color_by: 用於著色的數據（可選）
            cmap: 色彩映射

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 稀疏化數據
        x_sparse = x[::skip, ::skip]
        y_sparse = y[::skip, ::skip]
        u_sparse = u[::skip, ::skip]
        v_sparse = v[::skip, ::skip]

        # 計算顏色
        if color_by is not None:
            c = color_by[::skip, ::skip]
        else:
            c = np.sqrt(u_sparse**2 + v_sparse**2)

        # 繪製向量
        q = ax.quiver(
            x_sparse,
            y_sparse,
            u_sparse,
            v_sparse,
            c,
            cmap=cmap,
            scale=scale,
        )
        fig.colorbar(q, ax=ax, shrink=0.8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.set_aspect("equal")

        plt.tight_layout()
        return fig

    def plot_meridional(
        self,
        x: NDArray[np.floating],
        r: NDArray[np.floating],
        data: NDArray[np.floating],
        title: str = "子午面流場",
        cmap: str = "jet",
        levels: int = 20,
        colorbar_label: str = "",
        hub_x: NDArray[np.floating] | None = None,
        hub_r: NDArray[np.floating] | None = None,
        shroud_x: NDArray[np.floating] | None = None,
        shroud_r: NDArray[np.floating] | None = None,
    ) -> Figure:
        """繪製子午面流場。

        Args:
            x: 軸向座標 (2D array)
            r: 半徑座標 (2D array)
            data: 數據場 (2D array)
            title: 圖標題
            cmap: 色彩映射
            levels: 等值線數量
            colorbar_label: 色標標籤
            hub_x: 輪轂 x 座標（可選）
            hub_r: 輪轂半徑（可選）
            shroud_x: 機匣 x 座標（可選）
            shroud_r: 機匣半徑（可選）

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 繪製等值線
        cf = ax.contourf(x, r, data, levels=levels, cmap=cmap)
        ax.contour(x, r, data, levels=levels, colors="k", linewidths=0.3, alpha=0.5)

        # 添加色標
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
        if colorbar_label:
            cbar.set_label(colorbar_label)

        # 繪製輪轂和機匣
        if hub_x is not None and hub_r is not None:
            ax.plot(hub_x, hub_r, "k-", lw=2, label="輪轂")
        if shroud_x is not None and shroud_r is not None:
            ax.plot(shroud_x, shroud_r, "k-", lw=2, label="機匣")

        ax.set_xlabel("X (軸向)")
        ax.set_ylabel("R (半徑)")
        ax.set_title(title)

        if hub_x is not None or shroud_x is not None:
            ax.legend(loc="best")

        plt.tight_layout()
        return fig


def plot_contour(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    data: NDArray[np.floating],
    title: str = "流場等值線",
    cmap: str = "jet",
    levels: int = 20,
    colorbar_label: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製等值線圖的便捷函數。

    Args:
        x: x 座標 (2D array)
        y: y 座標 (2D array)
        data: 數據場 (2D array)
        title: 圖標題
        cmap: 色彩映射
        levels: 等值線數量
        colorbar_label: 色標標籤
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = FlowPlotter()
    fig = plotter.plot_contour(x, y, data, title, cmap, levels, colorbar_label)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_streamlines(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    title: str = "流線圖",
    density: float = 1.5,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """繪製流線圖的便捷函數。

    Args:
        x: x 座標 (1D or 2D array)
        y: y 座標 (1D or 2D array)
        u: x 方向速度 (2D array)
        v: y 方向速度 (2D array)
        title: 圖標題
        density: 流線密度
        save_path: 儲存路徑（可選）
        show: 是否顯示圖形

    Returns:
        matplotlib Figure 物件
    """
    plotter = FlowPlotter()
    fig = plotter.plot_streamlines(x, y, u, v, title, density)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
