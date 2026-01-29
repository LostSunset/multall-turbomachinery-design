# -*- coding: utf-8 -*-
"""STAGEN 流線投影測試。"""

from __future__ import annotations

import math

import pytest

from multall_turbomachinery_design.stagen.data_structures import (
    BladeProfile2D,
    StackingParameters,
    StreamSurface3D,
)
from multall_turbomachinery_design.stagen.projection import StreamSurfaceProjector


@pytest.fixture
def projector() -> StreamSurfaceProjector:
    """創建流線投影器。"""
    return StreamSurfaceProjector()


@pytest.fixture
def simple_stream_surface() -> StreamSurface3D:
    """創建簡單的流線表面（軸向流）。"""
    # 軸向流：半徑恆定
    x_coords = [0.0, 0.025, 0.05, 0.075, 0.10]
    r_coords = [0.3, 0.3, 0.3, 0.3, 0.3]

    projector = StreamSurfaceProjector()
    return projector.create_stream_surface(x_coords, r_coords)


@pytest.fixture
def simple_profile() -> BladeProfile2D:
    """創建簡單的 2D 葉片截面。"""
    # 單位弦長的葉片
    n = 50
    x = [i / (n - 1) for i in range(n)]
    y = [0.0] * n
    tk = [0.02] * n

    return BladeProfile2D(
        x_camber=x,
        y_camber=y,
        thickness=tk,
        chord_length=1.0,
    )


def test_projector_initialization(projector: StreamSurfaceProjector) -> None:
    """測試投影器初始化。"""
    assert projector is not None


def test_create_stream_surface_basic(projector: StreamSurfaceProjector) -> None:
    """測試基本流線表面創建。"""
    x_coords = [0.0, 0.05, 0.10]
    r_coords = [0.3, 0.3, 0.3]

    surface = projector.create_stream_surface(x_coords, r_coords)

    assert surface.npoints == 3
    assert surface.x == x_coords
    assert surface.r == r_coords


def test_meridional_distance_axial_flow(projector: StreamSurfaceProjector) -> None:
    """測試軸向流的子午線距離（應該等於軸向距離）。"""
    x_coords = [0.0, 0.025, 0.05, 0.075, 0.10]
    r_coords = [0.3, 0.3, 0.3, 0.3, 0.3]

    surface = projector.create_stream_surface(x_coords, r_coords)

    # 軸向流：子午線距離 = 軸向距離
    assert surface.s_meridional[0] == pytest.approx(0.0)
    assert surface.s_meridional[1] == pytest.approx(0.025)
    assert surface.s_meridional[2] == pytest.approx(0.05)
    assert surface.s_meridional[3] == pytest.approx(0.075)
    assert surface.s_meridional[4] == pytest.approx(0.10)


def test_meridional_distance_radial_flow(projector: StreamSurfaceProjector) -> None:
    """測試徑向流的子午線距離。"""
    x_coords = [0.0, 0.0, 0.0, 0.0]
    r_coords = [0.2, 0.25, 0.3, 0.35]

    surface = projector.create_stream_surface(x_coords, r_coords)

    # 徑向流：子午線距離 = 徑向距離
    assert surface.s_meridional[0] == pytest.approx(0.0)
    assert surface.s_meridional[1] == pytest.approx(0.05)
    assert surface.s_meridional[2] == pytest.approx(0.10)
    assert surface.s_meridional[3] == pytest.approx(0.15)


def test_meridional_distance_mixed_flow(projector: StreamSurfaceProjector) -> None:
    """測試混流的子午線距離。"""
    x_coords = [0.0, 0.03, 0.06]
    r_coords = [0.3, 0.32, 0.34]

    surface = projector.create_stream_surface(x_coords, r_coords)

    # 混流：子午線距離 = sqrt(dx^2 + dr^2)
    ds1 = math.sqrt(0.03**2 + 0.02**2)
    ds2 = math.sqrt(0.03**2 + 0.02**2)

    assert surface.s_meridional[0] == pytest.approx(0.0)
    assert surface.s_meridional[1] == pytest.approx(ds1, rel=1e-6)
    assert surface.s_meridional[2] == pytest.approx(ds1 + ds2, rel=1e-6)


def test_locate_leading_trailing_edges(
    projector: StreamSurfaceProjector, simple_stream_surface: StreamSurface3D
) -> None:
    """測試定位前後緣。"""
    surface = simple_stream_surface

    le_x = 0.02
    te_x = 0.08

    projector.locate_leading_trailing_edges(surface, le_x, te_x)

    # 檢查前後緣軸向坐標
    assert surface.le_x == le_x
    assert surface.te_x == te_x

    # 檢查半徑（軸向流應該相同）
    assert surface.le_r == pytest.approx(0.3)
    assert surface.te_r == pytest.approx(0.3)

    # 檢查子午線距離
    assert surface.le_s == pytest.approx(0.02)
    assert surface.te_s == pytest.approx(0.08)

    # 檢查子午弦長
    assert surface.chord_meridional == pytest.approx(0.06)


def test_project_profile_to_surface(
    projector: StreamSurfaceProjector,
    simple_profile: BladeProfile2D,
    simple_stream_surface: StreamSurface3D,
) -> None:
    """測試將 2D 截面投影到 3D 流線表面。"""
    # 先定位前後緣
    projector.locate_leading_trailing_edges(simple_stream_surface, 0.02, 0.08)

    # 投影
    section = projector.project_profile_to_surface(
        profile=simple_profile,
        surface=simple_stream_surface,
        section_number=1,
        spanwise_fraction=0.0,
    )

    # 檢查截面資訊
    assert section.section_number == 1
    assert section.spanwise_fraction == 0.0

    # 檢查點數
    assert len(section.x_grid) == len(simple_profile.x_camber)
    assert len(section.y_grid) == len(simple_profile.y_camber)
    assert len(section.r_grid) == len(simple_profile.x_camber)
    assert len(section.tk_grid) == len(simple_profile.thickness)

    # 檢查前後緣索引
    assert section.j_le == 0
    assert section.j_te == len(simple_profile.x_camber) - 1


def test_projection_scaling(
    projector: StreamSurfaceProjector,
    simple_profile: BladeProfile2D,
    simple_stream_surface: StreamSurface3D,
) -> None:
    """測試投影的縮放因子。"""
    # 單位弦長投影到子午弦長
    projector.locate_leading_trailing_edges(simple_stream_surface, 0.02, 0.08)

    section = projector.project_profile_to_surface(
        profile=simple_profile,
        surface=simple_stream_surface,
        section_number=1,
        spanwise_fraction=0.0,
    )

    # 子午弦長為 0.06
    # 單位弦長為 1.0
    # 縮放因子為 0.06
    scale = 0.06

    # 檢查厚度縮放
    for i, tk_orig in enumerate(simple_profile.thickness):
        assert section.tk_grid[i] == pytest.approx(tk_orig * scale)


def test_calculate_centroid_simple(
    projector: StreamSurfaceProjector,
) -> None:
    """測試簡單幾何的質心計算。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    # 創建矩形截面（厚度恆定）
    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 2

    projector.calculate_centroid(section)

    # 矩形質心應該在中點
    assert section.x_centroid == pytest.approx(0.025, rel=0.1)
    assert section.y_centroid == pytest.approx(0.0, abs=0.01)


def test_calculate_centroid_zero_area(
    projector: StreamSurfaceProjector,
) -> None:
    """測試零面積的質心計算。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    # 零厚度截面
    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.05, 0.10],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.0, 0.0, 0.0],
    )
    section.j_le = 0
    section.j_te = 2

    projector.calculate_centroid(section)

    # 零面積應該使用中點
    assert section.x_centroid == pytest.approx(0.05)
    assert section.y_centroid == pytest.approx(0.0)


def test_convert_r_theta_to_cartesian(
    projector: StreamSurfaceProjector,
) -> None:
    """測試 R-THETA 到笛卡爾坐標的轉換。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    # 創建簡單截面
    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.025, 0.05, 0.075, 0.10],
        y_grid=[0.0, 0.005, 0.010, 0.005, 0.0],
        r_grid=[0.3, 0.3, 0.3, 0.3, 0.3],
        tk_grid=[0.02, 0.02, 0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 4

    # 計算質心
    projector.calculate_centroid(section)

    # 轉換坐標
    projector.convert_r_theta_to_cartesian(section)

    # 檢查質心相對坐標（應該接近 0）
    mid_idx = len(section.x_grid) // 2
    y_at_centroid = section.y_grid[mid_idx]
    assert abs(y_at_centroid) < 0.01  # 質心附近的 Y 應該接近 0


def test_apply_stacking_no_transformation(
    projector: StreamSurfaceProjector,
    simple_stream_surface: StreamSurface3D,
) -> None:
    """測試不應用堆疊變換（所有因子為 0）。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    # 創建截面
    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 2
    section.x_centroid = 0.025
    section.y_centroid = 0.0

    # 保存原始坐標
    x_orig = section.x_grid.copy()
    y_orig = section.y_grid.copy()

    # 不應用變換的堆疊參數
    stacking = StackingParameters(
        f_centroid=0.0,
        f_tang=0.0,
        f_lean=0.0,
        f_sweep=0.0,
        f_axial=0.0,
        f_scale=1.0,
    )

    projector.apply_stacking(section, stacking, simple_stream_surface)

    # 坐標應該基本不變（允許小的數值誤差）
    for i in range(len(x_orig)):
        assert section.x_grid[i] == pytest.approx(x_orig[i], abs=1e-9)
        assert section.y_grid[i] == pytest.approx(y_orig[i], abs=1e-9)


def test_apply_stacking_scale(
    projector: StreamSurfaceProjector,
    simple_stream_surface: StreamSurface3D,
) -> None:
    """測試縮放變換。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    section = BladeSection3D(
        section_number=1,
        spanwise_fraction=0.0,
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 2
    section.x_centroid = 0.025
    section.y_centroid = 0.0

    # 縮放 1.5 倍，固定點在前緣
    stacking = StackingParameters(
        f_scale=1.5,
        f_const=0.0,  # 固定前緣
    )

    projector.apply_stacking(section, stacking, simple_stream_surface)

    # 前緣應該不變
    assert section.x_grid[0] == pytest.approx(0.0)

    # 後緣應該縮放
    # 原始：0.05，固定點：0.0，縮放後：0.0 + 1.5*(0.05-0.0) = 0.075
    assert section.x_grid[2] == pytest.approx(0.075)


def test_apply_stacking_centroid_to_hub(
    projector: StreamSurfaceProjector,
    simple_stream_surface: StreamSurface3D,
) -> None:
    """測試向 HUB 中心堆疊。"""
    from multall_turbomachinery_design.stagen.data_structures import BladeSection3D

    section = BladeSection3D(
        section_number=2,  # 不是 HUB
        spanwise_fraction=0.5,
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.3, 0.3, 0.3],
        tk_grid=[0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 2
    section.x_centroid = 0.025
    section.y_centroid = 0.0

    # 向 HUB 中心堆疊
    stacking = StackingParameters(
        f_centroid=0.5,  # 50% 向 HUB
        x_centroid_hub=0.020,  # HUB 質心在 x=0.020
        y_centroid_hub=0.001,  # HUB 質心在 y=0.001
    )

    projector.apply_stacking(section, stacking, simple_stream_surface)

    # 質心應該移動
    expected_x_centroid = 0.025 + 0.5 * (0.020 - 0.025)
    expected_y_centroid = 0.0 + 0.5 * (0.001 - 0.0)

    assert section.x_centroid == pytest.approx(expected_x_centroid)
    assert section.y_centroid == pytest.approx(expected_y_centroid)
