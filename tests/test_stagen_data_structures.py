# -*- coding: utf-8 -*-
"""STAGEN 數據結構測試。"""

from __future__ import annotations

from multall_turbomachinery_design.stagen.data_structures import (
    BladeInputType,
    BladeProfile2D,
    BladeRow,
    BladeSection3D,
    GridParameters,
    StackingParameters,
    StreamSurface3D,
    ThicknessParameters,
)


def test_blade_input_type_enum() -> None:
    """測試葉片輸入類型枚舉。"""
    assert BladeInputType.DIRECT_COORDS.value == 0
    assert BladeInputType.CAMBER_THICKNESS.value == 1
    assert BladeInputType.SURFACE_THICKNESS.value == 2
    assert BladeInputType.SURFACE_SLOPES.value == 3
    assert BladeInputType.FLOW_ANGLES.value == 4


def test_blade_profile_2d_initialization() -> None:
    """測試 2D 葉片截面初始化。"""
    profile = BladeProfile2D()

    assert profile.x_upper == []
    assert profile.y_upper == []
    assert profile.x_lower == []
    assert profile.y_lower == []
    assert profile.x_camber == []
    assert profile.y_camber == []
    assert profile.camber_slope == []
    assert profile.thickness == []
    assert profile.chord_length == 1.0
    assert profile.leading_edge_x == 0.0
    assert profile.trailing_edge_x == 1.0


def test_blade_profile_2d_with_data() -> None:
    """測試帶數據的 2D 葉片截面。"""
    profile = BladeProfile2D(
        x_upper=[0.0, 0.5, 1.0],
        y_upper=[0.0, 0.05, 0.0],
        x_lower=[0.0, 0.5, 1.0],
        y_lower=[0.0, -0.05, 0.0],
        chord_length=0.05,
    )

    assert len(profile.x_upper) == 3
    assert len(profile.y_upper) == 3
    assert profile.chord_length == 0.05


def test_stream_surface_3d_initialization() -> None:
    """測試 3D 流線表面初始化。"""
    surface = StreamSurface3D(npoints=10)

    assert surface.npoints == 10
    assert surface.x == []
    assert surface.r == []
    assert surface.s_meridional == []
    assert surface.le_x == 0.0
    assert surface.le_r == 0.0
    assert surface.te_x == 0.0
    assert surface.te_r == 0.0
    assert surface.chord_meridional == 0.0


def test_stream_surface_3d_with_data() -> None:
    """測試帶數據的流線表面。"""
    surface = StreamSurface3D(
        npoints=3,
        x=[0.0, 0.05, 0.10],
        r=[0.3, 0.3, 0.3],
        s_meridional=[0.0, 0.05, 0.10],
    )

    assert surface.npoints == 3
    assert len(surface.x) == 3
    assert len(surface.r) == 3
    assert len(surface.s_meridional) == 3


def test_stacking_parameters_defaults() -> None:
    """測試堆疊參數默認值。"""
    stacking = StackingParameters()

    assert stacking.f_centroid == 0.0
    assert stacking.f_tang == 0.0
    assert stacking.f_lean == 0.0
    assert stacking.f_sweep == 0.0
    assert stacking.f_axial == 0.0
    assert stacking.f_scale == 1.0
    assert stacking.f_const == 0.0
    assert stacking.x_centroid_hub == 0.0
    assert stacking.y_centroid_hub == 0.0


def test_stacking_parameters_custom() -> None:
    """測試自定義堆疊參數。"""
    stacking = StackingParameters(
        f_centroid=0.5,
        f_tang=0.1,
        f_lean=0.05,
        f_sweep=0.02,
        f_scale=1.1,
    )

    assert stacking.f_centroid == 0.5
    assert stacking.f_tang == 0.1
    assert stacking.f_lean == 0.05
    assert stacking.f_sweep == 0.02
    assert stacking.f_scale == 1.1


def test_blade_section_3d_initialization() -> None:
    """測試 3D 葉片截面初始化。"""
    section = BladeSection3D(section_number=1, spanwise_fraction=0.0)

    assert section.section_number == 1
    assert section.spanwise_fraction == 0.0
    assert section.x_grid == []
    assert section.y_grid == []
    assert section.r_grid == []
    assert section.s_grid == []
    assert section.tk_grid == []
    assert section.x_centroid == 0.0
    assert section.y_centroid == 0.0
    assert section.j_le == 0
    assert section.j_te == 0


def test_blade_section_3d_with_grid() -> None:
    """測試帶網格數據的 3D 葉片截面。"""
    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.005, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.002, 0.005, 0.001],
    )

    assert section.section_number == 1
    assert len(section.x_grid) == 3
    assert len(section.y_grid) == 3
    assert len(section.r_grid) == 3
    assert len(section.tk_grid) == 3


def test_thickness_parameters_defaults() -> None:
    """測試厚度參數默認值。"""
    params = ThicknessParameters()

    assert params.tk_le == 0.02
    assert params.tk_te == 0.01
    assert params.tk_max == 0.10
    assert params.xtk_max == 0.40
    assert params.tk_type == 2.0
    assert params.le_exp == 3.0
    assert params.xmod_le == 0.02
    assert params.xmod_te == 0.01
    assert params.f_perp == 1.0


def test_thickness_parameters_custom() -> None:
    """測試自定義厚度參數。"""
    params = ThicknessParameters(
        tk_le=0.03,
        tk_te=0.015,
        tk_max=0.12,
        xtk_max=0.35,
    )

    assert params.tk_le == 0.03
    assert params.tk_te == 0.015
    assert params.tk_max == 0.12
    assert params.xtk_max == 0.35


def test_grid_parameters_defaults() -> None:
    """測試網格參數默認值。"""
    grid = GridParameters()

    assert grid.im == 37
    assert grid.km == 11
    assert grid.fp_rat == 1.25
    assert grid.fp_max == 20.0
    assert grid.fr_rat == 1.25
    assert grid.fr_max == 20.0
    assert grid.nint_up == 5
    assert grid.nint_on == 50
    assert grid.nint_dn == 10


def test_grid_parameters_custom() -> None:
    """測試自定義網格參數。"""
    grid = GridParameters(
        im=49,
        km=15,
        fp_rat=1.3,
        nint_on=100,
    )

    assert grid.im == 49
    assert grid.km == 15
    assert grid.fp_rat == 1.3
    assert grid.nint_on == 100


def test_blade_row_initialization() -> None:
    """測試葉片排初始化。"""
    row = BladeRow(
        row_number=1,
        row_type="R",
        n_blade=24,
        rpm=10000.0,
    )

    assert row.row_number == 1
    assert row.row_type == "R"
    assert row.n_blade == 24
    assert row.rpm == 10000.0
    assert row.sections == []
    assert row.grid_params is None
    assert row.j_le == 0
    assert row.j_te == 0
    assert row.j_m == 0


def test_blade_row_with_sections() -> None:
    """測試帶截面的葉片排。"""
    section1 = BladeSection3D(section_number=1, spanwise_fraction=0.0)
    section2 = BladeSection3D(section_number=2, spanwise_fraction=0.5)

    row = BladeRow(
        row_number=1,
        row_type="R",
        n_blade=24,
        sections=[section1, section2],
    )

    assert len(row.sections) == 2
    assert row.sections[0].section_number == 1
    assert row.sections[1].section_number == 2
