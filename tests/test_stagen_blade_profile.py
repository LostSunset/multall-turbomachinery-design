# -*- coding: utf-8 -*-
"""STAGEN 葉片截面生成器測試。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multall_turbomachinery_design.stagen.blade_profile import BladeProfileGenerator
from multall_turbomachinery_design.stagen.data_structures import ThicknessParameters


@pytest.fixture
def generator() -> BladeProfileGenerator:
    """創建葉片截面生成器。"""
    return BladeProfileGenerator()


@pytest.fixture
def simple_camber_data() -> tuple[list[float], list[float]]:
    """簡單的中弧線斜率數據。"""
    # 線性變化的斜率：從 0.1 到 -0.1
    x_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    camber_slope = [0.1, 0.05, 0.0, -0.05, -0.1]
    return camber_slope, x_fractions


@pytest.fixture
def default_thickness_params() -> ThicknessParameters:
    """默認厚度參數。"""
    return ThicknessParameters()


def test_generator_initialization(generator: BladeProfileGenerator) -> None:
    """測試生成器初始化。"""
    assert generator.pi == pytest.approx(math.pi, rel=1e-9)
    assert generator.deg2rad == pytest.approx(math.pi / 180, rel=1e-9)
    assert generator.rad2deg == pytest.approx(180 / math.pi, rel=1e-9)


def test_generate_from_camber_thickness_basic(
    generator: BladeProfileGenerator,
    simple_camber_data: tuple[list[float], list[float]],
    default_thickness_params: ThicknessParameters,
) -> None:
    """測試基本葉片截面生成。"""
    camber_slope, x_fractions = simple_camber_data

    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=default_thickness_params,
        npoints=50,
    )

    # 檢查生成的點數
    assert len(profile.x_camber) == 50
    assert len(profile.y_camber) == 50
    assert len(profile.thickness) == 50
    assert len(profile.x_upper) == 50
    assert len(profile.y_upper) == 50
    assert len(profile.x_lower) == 50
    assert len(profile.y_lower) == 50

    # 檢查中弧線範圍
    assert profile.x_camber[0] == pytest.approx(0.0)
    assert profile.x_camber[-1] == pytest.approx(1.0)

    # 檢查厚度為正
    assert all(tk >= 0 for tk in profile.thickness)


def test_camber_line_integration(
    generator: BladeProfileGenerator,
    simple_camber_data: tuple[list[float], list[float]],
    default_thickness_params: ThicknessParameters,
) -> None:
    """測試中弧線積分。"""
    camber_slope, x_fractions = simple_camber_data

    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=default_thickness_params,
        npoints=100,
    )

    # 前緣應該在原點
    assert profile.x_camber[0] == pytest.approx(0.0)
    assert profile.y_camber[0] == pytest.approx(0.0)

    # Y 坐標應該根據斜率變化
    # 正斜率 -> Y 增加
    # 負斜率 -> Y 減小
    mid_index = len(profile.y_camber) // 2
    assert profile.y_camber[mid_index] > 0  # 前半段為正斜率


def test_thickness_distribution(
    generator: BladeProfileGenerator,
) -> None:
    """測試厚度分佈計算。"""
    x = np.linspace(0.0, 1.0, 100)
    params = ThicknessParameters(
        tk_le=0.02,
        tk_te=0.01,
        tk_max=0.10,
        xtk_max=0.40,
    )

    thickness = generator._calculate_thickness_distribution(x, params)

    # 檢查前後緣厚度
    assert thickness[0] == pytest.approx(params.tk_le, rel=1e-6)
    assert thickness[-1] == pytest.approx(params.tk_te, rel=1e-6)

    # 最大厚度應該接近指定位置
    max_tk_index = np.argmax(thickness)
    max_tk_x = x[max_tk_index]
    assert max_tk_x == pytest.approx(params.xtk_max, abs=0.05)

    # 最大厚度應該接近指定值
    assert max(thickness) == pytest.approx(params.tk_max, rel=0.1)


def test_thickness_distribution_power_transformation(
    generator: BladeProfileGenerator,
) -> None:
    """測試厚度分佈的功率變換。"""
    x = np.linspace(0.0, 1.0, 100)

    # 測試不同的最大厚度位置
    for xtk_max in [0.3, 0.4, 0.5, 0.6]:
        params = ThicknessParameters(
            tk_le=0.02,
            tk_te=0.01,
            tk_max=0.10,
            xtk_max=xtk_max,
        )

        thickness = generator._calculate_thickness_distribution(x, params)
        max_tk_index = np.argmax(thickness)
        max_tk_x = x[max_tk_index]

        # 最大厚度位置應該接近指定值
        assert max_tk_x == pytest.approx(xtk_max, abs=0.05)


def test_le_te_factor(
    generator: BladeProfileGenerator,
) -> None:
    """測試前後緣調整因子。"""
    x = np.linspace(0.0, 1.0, 100)
    params = ThicknessParameters(
        xmod_le=0.02,
        xmod_te=0.01,
        le_exp=3.0,
    )

    faclete = generator._calculate_le_te_factor(x, params)

    # 所有因子應該在 0 到 1 之間
    assert all(0 <= f <= 1 for f in faclete)

    # 中間部分應該接近 1
    mid_start = len(faclete) // 4
    mid_end = 3 * len(faclete) // 4
    assert all(f > 0.99 for f in faclete[mid_start:mid_end])

    # 前後緣應該小於 1
    assert faclete[0] < 1.0
    assert faclete[-1] < 1.0


def test_add_thickness_to_camber(
    generator: BladeProfileGenerator,
) -> None:
    """測試添加厚度到中弧線。"""
    # 直線中弧線
    n = 50
    x_camber = np.linspace(0.0, 1.0, n)
    y_camber = np.zeros(n)
    slope = np.zeros(n)
    thickness = np.full(n, 0.02)

    params = ThicknessParameters(f_perp=1.0)

    x_upper, y_upper, x_lower, y_lower = generator._add_thickness_to_camber(
        x_camber, y_camber, slope, thickness, params
    )

    # 對於水平中弧線，上下表面應該在 Y 方向對稱
    assert len(x_upper) == n
    assert len(y_upper) == n
    assert len(x_lower) == n
    assert len(y_lower) == n

    # 上表面在上方
    assert all(y_upper[i] >= y_camber[i] for i in range(n))
    # 下表面在下方
    assert all(y_lower[i] <= y_camber[i] for i in range(n))

    # 上下表面之間的距離應該接近厚度
    for i in range(n):
        distance = y_upper[i] - y_lower[i]
        assert distance == pytest.approx(thickness[i], rel=0.01)


def test_smooth_curve(generator: BladeProfileGenerator) -> None:
    """測試曲線平滑。"""
    # 創建帶噪聲的數據
    y_noisy = [0.0, 0.1, 0.15, 0.12, 0.2, 0.25, 0.22, 0.3]

    # 平滑
    y_smooth = generator.smooth_curve(y_noisy, n_smooth=3, smooth_factor=0.5)

    assert len(y_smooth) == len(y_noisy)

    # 保持端點
    assert y_smooth[0] == pytest.approx(y_noisy[0], abs=0.01)
    assert y_smooth[-1] == pytest.approx(y_noisy[-1], abs=0.01)

    # 平滑後的曲線應該更平順（曲率變化減小）
    # 計算二階差分（曲率的近似）
    def curvature_measure(y_vals: list[float]) -> float:
        y_arr = np.array(y_vals)
        diff2 = np.abs(np.diff(y_arr, n=2))
        return float(np.sum(diff2))

    curvature_original = curvature_measure(y_noisy)
    curvature_smoothed = curvature_measure(y_smooth)

    # 平滑後曲率應該減小
    assert curvature_smoothed < curvature_original


def test_profile_chord_length_default(
    generator: BladeProfileGenerator,
    simple_camber_data: tuple[list[float], list[float]],
    default_thickness_params: ThicknessParameters,
) -> None:
    """測試默認弦長為1。"""
    camber_slope, x_fractions = simple_camber_data

    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=default_thickness_params,
    )

    # 默認弦長應該為 1
    assert profile.chord_length == 1.0


def test_profile_symmetry_with_zero_camber(
    generator: BladeProfileGenerator,
    default_thickness_params: ThicknessParameters,
) -> None:
    """測試零中弧線的對稱性。"""
    # 零斜率中弧線
    x_fractions = [0.0, 0.5, 1.0]
    camber_slope = [0.0, 0.0, 0.0]

    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=default_thickness_params,
        npoints=100,
    )

    # 中弧線應該保持在 Y=0
    assert all(abs(y) < 0.01 for y in profile.y_camber)

    # 上下表面應該對稱
    for i in range(len(profile.y_upper)):
        assert profile.y_upper[i] == pytest.approx(-profile.y_lower[i], abs=1e-6)
