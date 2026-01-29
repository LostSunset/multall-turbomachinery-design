# -*- coding: utf-8 -*-
"""STAGEN 主求解器測試。"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from multall_turbomachinery_design.stagen import (
    BladeProfile2D,
    BladeRow,
    BladeSection3D,
    GridParameters,
    StackingParameters,
    StagenConfig,
    StagenSolver,
    StreamSurface3D,
    ThicknessParameters,
    create_simple_blade_row,
)


class TestStagenSolverInit:
    """測試求解器初始化。"""

    def test_init_with_config(self) -> None:
        """測試從配置創建求解器。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=5,
            grid_params=GridParameters(),
        )

        solver = StagenSolver(config)

        assert solver.config is config
        assert solver.profile_gen is not None
        assert solver.projector is not None
        assert solver.grid_gen is not None


class TestStagenSolverBladeProfile:
    """測試葉片截面生成。"""

    def test_generate_blade_profile(self) -> None:
        """測試生成 2D 葉片截面。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=5,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        camber_slope = [0.15, 0.0, -0.15]
        x_fractions = [0.0, 0.5, 1.0]
        thickness_params = ThicknessParameters(
            tk_le=0.02,
            tk_te=0.01,
            tk_max=0.10,
            xtk_max=0.40,
        )

        profile = solver.generate_blade_profile(
            camber_slope=camber_slope,
            x_fractions=x_fractions,
            thickness_params=thickness_params,
            npoints=100,
        )

        assert isinstance(profile, BladeProfile2D)
        assert len(profile.x_camber) == 100
        assert len(profile.y_camber) == 100
        assert len(profile.thickness) == 100
        assert profile.chord_length == pytest.approx(1.0, rel=0.01)


class TestStagenSolverStreamSurface:
    """測試流線表面創建。"""

    def test_create_stream_surface(self) -> None:
        """測試創建流線表面。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=5,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        x_coords = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        r_coords = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30]

        surface = solver.create_stream_surface(
            x_coords=x_coords,
            r_coords=r_coords,
            le_x=0.02,
            te_x=0.08,
        )

        assert isinstance(surface, StreamSurface3D)
        assert surface.le_x == pytest.approx(0.02)
        assert surface.te_x == pytest.approx(0.08)
        assert surface.chord_meridional == pytest.approx(0.06, rel=0.01)


class TestStagenSolverProjection:
    """測試 3D 投影。"""

    def test_project_to_3d(self) -> None:
        """測試將 2D 葉片投影到 3D。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=5,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        # 生成 2D 輪廓
        profile = solver.generate_blade_profile(
            camber_slope=[0.1, 0.0, -0.1],
            x_fractions=[0.0, 0.5, 1.0],
            thickness_params=ThicknessParameters(tk_max=0.08),
        )

        # 創建流線表面
        surface = solver.create_stream_surface(
            x_coords=[0.0, 0.025, 0.05, 0.075, 0.10],
            r_coords=[0.30, 0.30, 0.30, 0.30, 0.30],
            le_x=0.02,
            te_x=0.08,
        )

        # 投影到 3D
        section = solver.project_to_3d(
            profile=profile,
            surface=surface,
            section_number=1,
            spanwise_fraction=0.0,
        )

        assert isinstance(section, BladeSection3D)
        assert section.section_number == 1
        assert section.spanwise_fraction == 0.0
        assert len(section.x_grid) > 0
        assert len(section.y_grid) == len(section.x_grid)
        assert len(section.r_grid) == len(section.x_grid)


class TestStagenSolverStacking:
    """測試堆疊變換。"""

    def test_apply_stacking(self) -> None:
        """測試應用堆疊變換。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=5,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        # 創建一個簡單的截面
        section = BladeSection3D(
            section_number=2,
            spanwise_fraction=0.5,
            x_grid=[0.0, 0.025, 0.05],
            y_grid=[0.0, 0.0, 0.0],
            r_grid=[0.35, 0.35, 0.35],
            tk_grid=[0.02, 0.02, 0.02],
        )
        section.j_le = 0
        section.j_te = 2
        section.x_centroid = 0.025
        section.y_centroid = 0.0

        # 創建流線表面
        surface = solver.create_stream_surface(
            x_coords=[0.0, 0.025, 0.05, 0.075, 0.10],
            r_coords=[0.35, 0.35, 0.35, 0.35, 0.35],
            le_x=0.0,
            te_x=0.05,
        )

        # 定義堆疊參數
        stacking = StackingParameters(
            f_tang=0.1,
            f_lean=0.05,
            f_scale=1.0,
        )

        # 記錄原始坐標
        original_y = section.y_grid.copy()

        # 應用堆疊
        solver.apply_stacking(section, stacking, surface)

        # 檢查坐標已變化
        assert section.y_grid != original_y


class TestStagenSolverBladeRow:
    """測試葉片排生成。"""

    def test_generate_blade_row(self) -> None:
        """測試生成葉片排。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=3,
            grid_params=GridParameters(km=3, fr_rat=1.0, fr_max=1.0),
        )
        solver = StagenSolver(config)

        # 生成三個截面的輪廓和表面
        profiles = []
        surfaces = []

        for i in range(3):
            profile = solver.generate_blade_profile(
                camber_slope=[0.1, 0.0, -0.1],
                x_fractions=[0.0, 0.5, 1.0],
                thickness_params=ThicknessParameters(tk_max=0.08),
            )
            profiles.append(profile)

            r = 0.25 + i * 0.05  # HUB 到 TIP
            surface = solver.create_stream_surface(
                x_coords=[0.0, 0.025, 0.05, 0.075, 0.10],
                r_coords=[r, r, r, r, r],
                le_x=0.02,
                te_x=0.08,
            )
            surfaces.append(surface)

        # 生成葉片排
        blade_row = solver.generate_blade_row(
            row_number=1,
            row_type="R",
            n_blades=30,
            rpm=3000.0,
            profiles=profiles,
            surfaces=surfaces,
        )

        assert isinstance(blade_row, BladeRow)
        assert blade_row.row_number == 1
        assert blade_row.row_type == "R"
        assert blade_row.n_blade == 30
        assert blade_row.rpm == 3000.0
        assert len(blade_row.sections) == 3


class TestStagenSolverSolveBladeRow:
    """測試完整葉片排求解。"""

    def test_solve_blade_row(self) -> None:
        """測試求解葉片排。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=3,
            grid_params=GridParameters(km=3, fr_rat=1.0, fr_max=1.0),
        )
        solver = StagenSolver(config)

        # 流線坐標
        x_coords = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        r_hub = [0.25] * 6
        r_tip = [0.35] * 6

        # 中弧線斜率
        inlet_angle = 30.0
        outlet_angle = -30.0
        deg2rad = math.pi / 180.0
        inlet_slope = math.tan(inlet_angle * deg2rad)
        outlet_slope = math.tan(outlet_angle * deg2rad)
        camber_slopes = [[inlet_slope, 0.0, outlet_slope]]
        x_fractions = [0.0, 0.5, 1.0]

        # 厚度參數
        thickness_params = ThicknessParameters(
            tk_le=0.02,
            tk_te=0.01,
            tk_max=0.08,
            xtk_max=0.40,
        )

        # 求解葉片排
        blade_row = solver.solve_blade_row(
            row_number=1,
            row_type="R",
            n_blades=30,
            rpm=3000.0,
            x_hub=x_coords,
            r_hub=r_hub,
            x_tip=x_coords,
            r_tip=r_tip,
            le_x=0.02,
            te_x=0.08,
            camber_slopes=camber_slopes,
            x_fractions=x_fractions,
            thickness_params=thickness_params,
        )

        assert isinstance(blade_row, BladeRow)
        assert len(blade_row.sections) == 3

        # 檢查截面的跨向分數
        for i, section in enumerate(blade_row.sections):
            assert section.section_number == i + 1
            assert 0.0 <= section.spanwise_fraction <= 1.0


class TestStagenSolverOutput:
    """測試輸出功能。"""

    def test_write_outputs(self) -> None:
        """測試寫入輸出文件。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=1,
            nosect=3,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            solver.write_outputs(tmpdir)

            # 檢查輸出文件
            assert (Path(tmpdir) / "stage_old.dat").exists()
            assert (Path(tmpdir) / "stage_new.dat").exists()
            assert (Path(tmpdir) / "stagen.out").exists()

    def test_run_with_output(self) -> None:
        """測試完整運行並輸出。"""
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=0,
            nosect=3,
            grid_params=GridParameters(),
        )
        solver = StagenSolver(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = solver.run(output_dir=tmpdir)

            assert "nrows" in result
            assert "nosect" in result
            assert "output_dir" in result
            assert result["output_dir"] == tmpdir


class TestCreateSimpleBladeRow:
    """測試便捷函數。"""

    def test_create_simple_rotor(self) -> None:
        """測試創建簡單轉子。"""
        solver = create_simple_blade_row(
            row_type="R",
            n_blades=30,
            rpm=3000.0,
            r_hub=0.25,
            r_tip=0.35,
            le_x=0.02,
            te_x=0.08,
            inlet_angle=30.0,
            outlet_angle=-30.0,
            tk_max=0.08,
            nosect=3,
        )

        assert isinstance(solver, StagenSolver)
        assert len(solver.config.blade_rows) == 1

        blade_row = solver.config.blade_rows[0]
        assert blade_row.row_type == "R"
        assert blade_row.n_blade == 30
        assert blade_row.rpm == 3000.0
        assert len(blade_row.sections) == 3

    def test_create_simple_stator(self) -> None:
        """測試創建簡單定子。"""
        solver = create_simple_blade_row(
            row_type="S",
            n_blades=40,
            rpm=0.0,
            r_hub=0.25,
            r_tip=0.35,
            le_x=0.10,
            te_x=0.16,
            inlet_angle=45.0,
            outlet_angle=0.0,
            tk_max=0.06,
            nosect=5,
        )

        assert isinstance(solver, StagenSolver)
        assert len(solver.config.blade_rows) == 1

        blade_row = solver.config.blade_rows[0]
        assert blade_row.row_type == "S"
        assert blade_row.n_blade == 40
        assert blade_row.rpm == 0.0
        assert len(blade_row.sections) == 5

    def test_create_and_output(self) -> None:
        """測試創建並輸出。"""
        solver = create_simple_blade_row(
            row_type="R",
            n_blades=30,
            rpm=3000.0,
            nosect=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = solver.run(output_dir=tmpdir)

            assert result["nrows"] == 1
            assert result["total_sections"] == 3
            assert (Path(tmpdir) / "stagen.out").exists()


class TestStagenSolverIntegration:
    """整合測試。"""

    def test_full_workflow(self) -> None:
        """測試完整工作流程。"""
        # 1. 創建配置
        config = StagenConfig(
            rgas=287.0,
            gamma=1.4,
            nrows=2,
            nosect=5,
            grid_params=GridParameters(
                im=37,
                km=5,
                fp_rat=1.25,
                fp_max=20.0,
                fr_rat=1.25,
                fr_max=20.0,
            ),
        )

        # 2. 創建求解器
        solver = StagenSolver(config)

        # 3. 定義流線坐標
        x_coords = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        r_hub = [0.25] * 6
        r_tip = [0.35] * 6

        # 4. 求解轉子
        rotor = solver.solve_blade_row(
            row_number=1,
            row_type="R",
            n_blades=30,
            rpm=3000.0,
            x_hub=x_coords,
            r_hub=r_hub,
            x_tip=x_coords,
            r_tip=r_tip,
            le_x=0.02,
            te_x=0.06,
            camber_slopes=[[0.5, 0.0, -0.5]],
            x_fractions=[0.0, 0.5, 1.0],
            thickness_params=ThicknessParameters(tk_max=0.08),
        )
        config.blade_rows.append(rotor)

        # 5. 求解定子
        stator = solver.solve_blade_row(
            row_number=2,
            row_type="S",
            n_blades=40,
            rpm=0.0,
            x_hub=[x + 0.04 for x in x_coords],
            r_hub=r_hub,
            x_tip=[x + 0.04 for x in x_coords],
            r_tip=r_tip,
            le_x=0.08,
            te_x=0.12,
            camber_slopes=[[0.3, 0.0, -0.1]],
            x_fractions=[0.0, 0.5, 1.0],
            thickness_params=ThicknessParameters(tk_max=0.06),
        )
        config.blade_rows.append(stator)

        # 6. 輸出結果
        with tempfile.TemporaryDirectory() as tmpdir:
            result = solver.run(output_dir=tmpdir)

            assert result["nrows"] == 2
            assert result["total_sections"] == 10  # 5 + 5

            # 檢查輸出文件
            assert (Path(tmpdir) / "stage_old.dat").exists()
            assert (Path(tmpdir) / "stage_new.dat").exists()
            assert (Path(tmpdir) / "stagen.out").exists()

            # 讀取並驗證輸出文件
            with open(Path(tmpdir) / "stagen.out", encoding="utf-8") as f:
                content = f.read()
                assert "STAGEN" in content
                assert "287" in content  # RGAS
                assert "1.4" in content  # GAMMA


class TestStagenSolverEdgeCases:
    """邊界情況測試。"""

    def test_single_section(self) -> None:
        """測試單截面葉片排。"""
        solver = create_simple_blade_row(nosect=1)
        assert len(solver.config.blade_rows[0].sections) == 1

    def test_many_sections(self) -> None:
        """測試多截面葉片排。"""
        solver = create_simple_blade_row(nosect=11)
        assert len(solver.config.blade_rows[0].sections) == 11

    def test_get_blade_coordinates(self) -> None:
        """測試獲取葉片坐標。"""
        solver = create_simple_blade_row(nosect=3)

        coords = solver.get_blade_coordinates(row_number=1)

        assert "x" in coords
        assert "y" in coords
        assert "r" in coords
        assert "thickness" in coords
        assert len(coords["x"]) == 3

    def test_get_blade_coordinates_invalid_row(self) -> None:
        """測試獲取不存在的排號。"""
        solver = create_simple_blade_row(nosect=3)

        with pytest.raises(ValueError, match="排號.*不存在"):
            solver.get_blade_coordinates(row_number=99)
