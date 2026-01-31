# -*- coding: utf-8 -*-
"""視覺化模組測試。"""

from __future__ import annotations

import numpy as np
import pytest

# 檢查 matplotlib 是否可用
try:
    import matplotlib

    matplotlib.use("Agg")  # 使用非互動式後端
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="需要 matplotlib",
)


class TestVelocityTrianglePlotter:
    """速度三角形繪圖器測試。"""

    def test_velocity_components(self) -> None:
        """測試速度分量計算。"""
        from multall_turbomachinery_design.visualization.velocity_triangle import (
            VelocityComponents,
        )

        vel = VelocityComponents(vx=100.0, vt=50.0, u=80.0)

        assert vel.vx == 100.0
        assert vel.vt == 50.0
        assert vel.u == 80.0
        assert np.isclose(vel.v, np.sqrt(100**2 + 50**2))
        assert np.isclose(vel.w, np.sqrt(100**2 + (50 - 80) ** 2))

    def test_velocity_angles(self) -> None:
        """測試速度角度計算。"""
        from multall_turbomachinery_design.visualization.velocity_triangle import (
            VelocityComponents,
        )

        vel = VelocityComponents(vx=100.0, vt=0.0, u=0.0)

        assert np.isclose(vel.alpha, 0.0)
        assert np.isclose(vel.beta, 0.0)

    def test_plot_creation(self) -> None:
        """測試繪圖創建。"""
        from multall_turbomachinery_design.visualization.velocity_triangle import (
            VelocityComponents,
            VelocityTrianglePlotter,
        )

        inlet = VelocityComponents(vx=150.0, vt=0.0, u=200.0)
        outlet = VelocityComponents(vx=150.0, vt=-100.0, u=200.0)

        plotter = VelocityTrianglePlotter()
        fig = plotter.plot(inlet, outlet)

        assert fig is not None
        plt.close(fig)

    def test_convenience_function(self) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.visualization import plot_velocity_triangle

        inlet = {"vx": 150.0, "vt": 0.0, "u": 200.0}
        outlet = {"vx": 150.0, "vt": -100.0, "u": 200.0}

        fig = plot_velocity_triangle(inlet, outlet, show=False)

        assert fig is not None
        plt.close(fig)


class TestBladePlotter:
    """葉片繪圖器測試。"""

    @pytest.fixture
    def blade_data(self) -> dict:
        """創建葉片數據。"""
        t = np.linspace(0, 1, 50)
        x = t
        y_upper = 0.05 * np.sin(np.pi * t)
        y_lower = -0.03 * np.sin(np.pi * t)
        y_camber = (y_upper + y_lower) / 2

        return {
            "x_upper": x,
            "y_upper": y_upper,
            "x_lower": x,
            "y_lower": y_lower,
            "x_camber": x,
            "y_camber": y_camber,
        }

    def test_plot_profile(self, blade_data: dict) -> None:
        """測試葉片截面繪圖。"""
        from multall_turbomachinery_design.visualization.blade_plots import BladePlotter

        plotter = BladePlotter()
        fig = plotter.plot_profile(**blade_data)

        assert fig is not None
        plt.close(fig)

    def test_plot_cascade(self, blade_data: dict) -> None:
        """測試葉柵繪圖。"""
        from multall_turbomachinery_design.visualization.blade_plots import BladePlotter

        plotter = BladePlotter()
        fig = plotter.plot_cascade(
            blade_data["x_upper"],
            blade_data["y_upper"],
            blade_data["x_lower"],
            blade_data["y_lower"],
            pitch=0.08,
            n_blades=3,
        )

        assert fig is not None
        plt.close(fig)

    def test_convenience_function(self, blade_data: dict) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.visualization import plot_blade_profile

        fig = plot_blade_profile(**blade_data, show=False)

        assert fig is not None
        plt.close(fig)


class TestFlowPlotter:
    """流場繪圖器測試。"""

    @pytest.fixture
    def flow_data(self) -> dict:
        """創建流場數據。"""
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 40)
        x_grid, y_grid = np.meshgrid(x, y)

        # 簡單的流場
        u = np.ones_like(x_grid) * 100
        v = np.sin(np.pi * y_grid) * 10
        p = 101325 - 1000 * x_grid

        return {"x": x_grid, "y": y_grid, "u": u, "v": v, "p": p}

    def test_plot_contour(self, flow_data: dict) -> None:
        """測試等值線繪圖。"""
        from multall_turbomachinery_design.visualization.flow_plots import FlowPlotter

        plotter = FlowPlotter()
        fig = plotter.plot_contour(
            flow_data["x"],
            flow_data["y"],
            flow_data["p"],
            title="壓力分佈",
            colorbar_label="P (Pa)",
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_streamlines(self, flow_data: dict) -> None:
        """測試流線繪圖。"""
        from multall_turbomachinery_design.visualization.flow_plots import FlowPlotter

        plotter = FlowPlotter()
        fig = plotter.plot_streamlines(
            flow_data["x"],
            flow_data["y"],
            flow_data["u"],
            flow_data["v"],
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_vectors(self, flow_data: dict) -> None:
        """測試向量場繪圖。"""
        from multall_turbomachinery_design.visualization.flow_plots import FlowPlotter

        plotter = FlowPlotter()
        fig = plotter.plot_vectors(
            flow_data["x"],
            flow_data["y"],
            flow_data["u"],
            flow_data["v"],
            skip=3,
        )

        assert fig is not None
        plt.close(fig)

    def test_convenience_functions(self, flow_data: dict) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.visualization import (
            plot_contour,
            plot_streamlines,
        )

        fig1 = plot_contour(
            flow_data["x"],
            flow_data["y"],
            flow_data["p"],
            show=False,
        )
        assert fig1 is not None
        plt.close(fig1)

        fig2 = plot_streamlines(
            flow_data["x"],
            flow_data["y"],
            flow_data["u"],
            flow_data["v"],
            show=False,
        )
        assert fig2 is not None
        plt.close(fig2)


class TestConvergencePlotter:
    """收斂歷史繪圖器測試。"""

    @pytest.fixture
    def convergence_data(self) -> dict:
        """創建收斂數據。"""
        iterations = np.arange(1, 101)
        residual = 1.0 * np.exp(-0.05 * iterations)
        efficiency = 0.85 + 0.05 * (1 - np.exp(-0.1 * iterations))
        mass_flow = 10.0 + 0.1 * np.sin(0.1 * iterations) * np.exp(-0.05 * iterations)

        return {
            "iterations": iterations,
            "residual": residual,
            "efficiency": efficiency,
            "mass_flow": mass_flow,
        }

    def test_plot_residuals(self, convergence_data: dict) -> None:
        """測試殘差繪圖。"""
        from multall_turbomachinery_design.visualization.convergence_plots import (
            ConvergencePlotter,
        )

        plotter = ConvergencePlotter()
        fig = plotter.plot_residuals(
            convergence_data["iterations"],
            {"殘差": convergence_data["residual"]},
            target_residual=1e-4,
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_performance(self, convergence_data: dict) -> None:
        """測試性能繪圖。"""
        from multall_turbomachinery_design.visualization.convergence_plots import (
            ConvergencePlotter,
        )

        plotter = ConvergencePlotter()
        fig = plotter.plot_performance(
            convergence_data["iterations"],
            {
                "效率": convergence_data["efficiency"],
                "質量流率": convergence_data["mass_flow"],
            },
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_combined(self, convergence_data: dict) -> None:
        """測試組合繪圖。"""
        from multall_turbomachinery_design.visualization.convergence_plots import (
            ConvergencePlotter,
        )

        plotter = ConvergencePlotter()
        fig = plotter.plot_combined(
            convergence_data["iterations"],
            convergence_data["residual"],
            convergence_data["efficiency"],
            convergence_data["mass_flow"],
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_stage_performance(self) -> None:
        """測試各級性能繪圖。"""
        from multall_turbomachinery_design.visualization.convergence_plots import (
            ConvergencePlotter,
        )

        plotter = ConvergencePlotter()
        fig = plotter.plot_stage_performance(
            stages=[1, 2, 3],
            efficiency=[0.90, 0.89, 0.88],
            pressure_ratio=[1.8, 1.7, 1.6],
            temperature_ratio=[0.95, 0.94, 0.93],
        )

        assert fig is not None
        plt.close(fig)

    def test_convenience_functions(self, convergence_data: dict) -> None:
        """測試便捷函數。"""
        from multall_turbomachinery_design.visualization import (
            plot_convergence_history,
            plot_residual_history,
        )

        fig1 = plot_convergence_history(
            convergence_data["iterations"],
            convergence_data["residual"],
            convergence_data["efficiency"],
            convergence_data["mass_flow"],
            show=False,
        )
        assert fig1 is not None
        plt.close(fig1)

        fig2 = plot_residual_history(
            convergence_data["iterations"],
            {"殘差": convergence_data["residual"]},
            show=False,
        )
        assert fig2 is not None
        plt.close(fig2)


class TestModuleImports:
    """模組導入測試。"""

    def test_import_visualization(self) -> None:
        """測試導入視覺化模組。"""
        from multall_turbomachinery_design import visualization

        assert visualization is not None

    def test_import_all_plotters(self) -> None:
        """測試導入所有繪圖器。"""
        from multall_turbomachinery_design.visualization import (
            BladePlotter,
            ConvergencePlotter,
            FlowPlotter,
            VelocityTrianglePlotter,
        )

        assert BladePlotter is not None
        assert ConvergencePlotter is not None
        assert FlowPlotter is not None
        assert VelocityTrianglePlotter is not None

    def test_import_all_functions(self) -> None:
        """測試導入所有便捷函數。"""
        from multall_turbomachinery_design.visualization import (
            plot_blade_profile,
            plot_blade_surface,
            plot_contour,
            plot_convergence_history,
            plot_residual_history,
            plot_streamlines,
            plot_velocity_triangle,
        )

        assert plot_blade_profile is not None
        assert plot_blade_surface is not None
        assert plot_contour is not None
        assert plot_convergence_history is not None
        assert plot_residual_history is not None
        assert plot_streamlines is not None
        assert plot_velocity_triangle is not None
