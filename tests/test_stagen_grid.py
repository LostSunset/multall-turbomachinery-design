# -*- coding: utf-8 -*-
"""STAGEN 網格生成器測試。"""

from __future__ import annotations

import pytest

from multall_turbomachinery_design.stagen.data_structures import (
    BladeRow,
    GridParameters,
)
from multall_turbomachinery_design.stagen.grid_generator import GridGenerator


@pytest.fixture
def generator() -> GridGenerator:
    """創建網格生成器。"""
    return GridGenerator()


@pytest.fixture
def default_grid_params() -> GridParameters:
    """默認網格參數。"""
    return GridParameters()


def test_generator_initialization(generator: GridGenerator) -> None:
    """測試網格生成器初始化。"""
    assert generator is not None


def test_calculate_pitchwise_expansion(generator: GridGenerator) -> None:
    """測試周向網格擴張計算。"""
    im = 37
    fp_rat = 1.25
    fp_max = 20.0

    fp = generator.calculate_pitchwise_expansion(im, fp_rat, fp_max)

    # 檢查長度
    assert len(fp) == im - 1

    # 檢查對稱性（應該從兩端向中心擴張）
    mid = len(fp) // 2
    # 左半部分應該遞增
    for i in range(mid - 1):
        assert fp[i + 1] >= fp[i]

    # 所有因子應該在 1.0 和 fp_max 之間
    assert all(1.0 <= f <= fp_max for f in fp)


def test_pitchwise_expansion_no_expansion(generator: GridGenerator) -> None:
    """測試無擴張的周向網格（均勻網格）。"""
    im = 20
    fp_rat = 1.0  # 無擴張
    fp_max = 20.0

    fp = generator.calculate_pitchwise_expansion(im, fp_rat, fp_max)

    # 所有因子應該為 1.0
    assert all(abs(f - 1.0) < 1e-9 for f in fp)


def test_pitchwise_expansion_max_limit(generator: GridGenerator) -> None:
    """測試周向網格擴張的最大限制。"""
    im = 50
    fp_rat = 2.0  # 高擴張比
    fp_max = 5.0  # 較低的最大限制

    fp = generator.calculate_pitchwise_expansion(im, fp_rat, fp_max)

    # 所有因子應該不超過 fp_max
    assert all(f <= fp_max for f in fp)


def test_calculate_spanwise_expansion(generator: GridGenerator) -> None:
    """測試跨向網格擴張計算。"""
    km = 11
    fr_rat = 1.25
    fr_max = 20.0

    fr, sumfr = generator.calculate_spanwise_expansion(km, fr_rat, fr_max)

    # 檢查長度
    assert len(fr) == km - 1
    assert len(sumfr) == km

    # 累積因子應該遞增
    for k in range(1, km):
        assert sumfr[k] >= sumfr[k - 1]

    # 第一個累積因子應該為 0
    assert sumfr[0] == 0.0

    # 所有因子應該在 1.0 和 fr_max 之間
    assert all(1.0 <= f <= fr_max for f in fr)


def test_spanwise_expansion_symmetry(generator: GridGenerator) -> None:
    """測試跨向網格擴張的對稱性。"""
    km = 15
    fr_rat = 1.3
    fr_max = 15.0

    fr, _sumfr = generator.calculate_spanwise_expansion(km, fr_rat, fr_max)

    # 從 HUB 和 CASING 兩端應該對稱擴張
    mid = len(fr) // 2
    for k in range(mid):
        # 左半部分遞增
        if k < mid - 1:
            assert fr[k + 1] >= fr[k]


def test_generate_pitchwise_grid(
    generator: GridGenerator, default_grid_params: GridParameters
) -> None:
    """測試周向網格生成。"""
    n_blades = 24
    pitch = 0.05  # 5 cm 間距
    im = default_grid_params.im

    y_grid = generator.generate_pitchwise_grid(n_blades, pitch, default_grid_params)

    # 檢查點數
    assert len(y_grid) == im

    # 網格應該中心化（質心接近 0）
    y_center = y_grid[im // 2]
    assert abs(y_center) < pitch / 10

    # 檢查範圍（應該在 pitch 範圍內）
    assert max(y_grid) - min(y_grid) <= pitch


def test_generate_pitchwise_grid_monotonic(
    generator: GridGenerator, default_grid_params: GridParameters
) -> None:
    """測試周向網格的單調性。"""
    n_blades = 30
    pitch = 0.04

    y_grid = generator.generate_pitchwise_grid(n_blades, pitch, default_grid_params)

    # 網格應該單調遞增
    for i in range(1, len(y_grid)):
        assert y_grid[i] > y_grid[i - 1]


def test_generate_spanwise_grid(
    generator: GridGenerator, default_grid_params: GridParameters
) -> None:
    """測試跨向網格生成。"""
    r_hub = 0.28
    r_tip = 0.32
    km = default_grid_params.km

    r_grid = generator.generate_spanwise_grid(r_hub, r_tip, default_grid_params)

    # 檢查點數
    assert len(r_grid) == km

    # 檢查邊界
    assert r_grid[0] == pytest.approx(r_hub)
    assert r_grid[-1] == pytest.approx(r_tip)

    # 檢查單調遞增
    for k in range(1, km):
        assert r_grid[k] > r_grid[k - 1]


def test_spanwise_grid_uniform_expansion(generator: GridGenerator) -> None:
    """測試均勻擴張的跨向網格。"""
    r_hub = 0.25
    r_tip = 0.35
    grid_params = GridParameters(km=11, fr_rat=1.0, fr_max=20.0)

    r_grid = generator.generate_spanwise_grid(r_hub, r_tip, grid_params)

    # 均勻擴張應該產生線性分佈
    dr = r_grid[1] - r_grid[0]
    for k in range(2, len(r_grid)):
        dr_k = r_grid[k] - r_grid[k - 1]
        assert dr_k == pytest.approx(dr, rel=1e-6)


def test_generate_axial_grid(generator: GridGenerator, default_grid_params: GridParameters) -> None:
    """測試軸向網格生成。"""
    x_le = 0.02
    x_te = 0.08
    axial_chord = 0.06

    x_grid, j_le, j_te = generator.generate_axial_grid(x_le, x_te, axial_chord, default_grid_params)

    # 檢查總點數
    expected_points = (
        default_grid_params.nint_up + default_grid_params.nint_on + default_grid_params.nint_dn + 1
    )
    assert len(x_grid) == expected_points

    # 檢查前後緣索引
    assert j_le == default_grid_params.nint_up
    assert j_te == default_grid_params.nint_up + default_grid_params.nint_on

    # 檢查前後緣坐標
    assert x_grid[j_le] == pytest.approx(x_le)
    assert x_grid[j_te] == pytest.approx(x_te)

    # 檢查單調遞增
    for j in range(1, len(x_grid)):
        assert x_grid[j] > x_grid[j - 1]


def test_axial_grid_coverage(generator: GridGenerator, default_grid_params: GridParameters) -> None:
    """測試軸向網格覆蓋範圍。"""
    x_le = 0.05
    x_te = 0.10
    axial_chord = 0.05

    x_grid, _j_le, _j_te = generator.generate_axial_grid(
        x_le, x_te, axial_chord, default_grid_params
    )

    # 檢查覆蓋範圍（上游 + 葉片 + 下游）
    assert x_grid[0] == pytest.approx(x_le - axial_chord)
    assert x_grid[-1] == pytest.approx(x_te + axial_chord)


def test_generate_blade_row_grid(generator: GridGenerator) -> None:
    """測試葉片排網格生成。"""
    # 創建葉片排
    blade_row = BladeRow(
        row_number=1,
        row_type="R",
        n_blade=24,
        rpm=10000.0,
    )
    blade_row.grid_params = GridParameters()

    # 幾何參數
    r_hub = 0.28
    r_tip = 0.32
    x_le = 0.02
    x_te = 0.08
    axial_chord = 0.06

    # 生成網格
    generator.generate_blade_row_grid(blade_row, r_hub, r_tip, x_le, x_te, axial_chord)

    # 檢查索引已設置
    assert blade_row.j_le > 0
    assert blade_row.j_te > blade_row.j_le
    assert blade_row.j_m > blade_row.j_te


def test_calculate_spanwise_fractions(generator: GridGenerator) -> None:
    """測試跨向分數位置計算。"""
    km = 11
    fr_rat = 1.25
    fr_max = 20.0

    fractions = generator.calculate_spanwise_fractions(km, fr_rat, fr_max)

    # 檢查長度
    assert len(fractions) == km

    # 檢查範圍
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[-1] == pytest.approx(1.0)

    # 檢查單調遞增
    for k in range(1, km):
        assert fractions[k] > fractions[k - 1]


def test_spanwise_fractions_uniform(generator: GridGenerator) -> None:
    """測試均勻跨向分數位置。"""
    km = 10
    fr_rat = 1.0  # 均勻
    fr_max = 20.0

    fractions = generator.calculate_spanwise_fractions(km, fr_rat, fr_max)

    # 均勻分佈應該線性
    for k in range(1, km):
        expected = k / (km - 1)
        assert fractions[k] == pytest.approx(expected, rel=1e-6)
