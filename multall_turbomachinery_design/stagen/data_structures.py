# -*- coding: utf-8 -*-
"""STAGEN 數據結構定義。

定義 3D 葉片幾何生成所需的核心數據結構。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BladeInputType(Enum):
    """葉片輸入類型。"""

    DIRECT_COORDS = 0  # 直接坐標輸入
    CAMBER_THICKNESS = 1  # 中弧線+厚度分佈
    SURFACE_THICKNESS = 2  # 表面斜率+上下厚度
    SURFACE_SLOPES = 3  # 上下表面斜率
    FLOW_ANGLES = 4  # 進出口流角


@dataclass
class BladeProfile2D:
    """2D 葉片截面輪廓。"""

    # 坐標數據
    x_upper: list[float] = field(default_factory=list)  # 上表面 X 坐標
    y_upper: list[float] = field(default_factory=list)  # 上表面 Y 坐標
    x_lower: list[float] = field(default_factory=list)  # 下表面 X 坐標
    y_lower: list[float] = field(default_factory=list)  # 下表面 Y 坐標

    # 中弧線數據
    x_camber: list[float] = field(default_factory=list)  # 中弧線 X 坐標
    y_camber: list[float] = field(default_factory=list)  # 中弧線 Y 坐標
    camber_slope: list[float] = field(default_factory=list)  # 中弧線斜率 dy/dx

    # 厚度數據
    thickness: list[float] = field(default_factory=list)  # 厚度分佈 [m]

    # 幾何參數
    chord_length: float = 1.0  # 弦長 [m]（默認單位弦）
    leading_edge_x: float = 0.0  # 前緣 X 坐標
    trailing_edge_x: float = 1.0  # 後緣 X 坐標


@dataclass
class StreamSurface3D:
    """3D 流線表面。"""

    npoints: int  # 點數
    x: list[float] = field(default_factory=list)  # 軸向坐標 [m]
    r: list[float] = field(default_factory=list)  # 半徑坐標 [m]
    s_meridional: list[float] = field(default_factory=list)  # 子午線距離 [m]

    # 前後緣位置
    le_x: float = 0.0  # 前緣軸向坐標 [m]
    le_r: float = 0.0  # 前緣半徑 [m]
    te_x: float = 0.0  # 後緣軸向坐標 [m]
    te_r: float = 0.0  # 後緣半徑 [m]

    # 子午線前後緣距離
    le_s: float = 0.0  # 前緣子午線距離 [m]
    te_s: float = 0.0  # 後緣子午線距離 [m]
    chord_meridional: float = 0.0  # 子午弦長 [m]


@dataclass
class StackingParameters:
    """3D 堆疊參數。"""

    # 堆疊因子（作為弦長的比例）
    f_centroid: float = 0.0  # 向 HUB 中心堆疊因子 (0-1)
    f_tang: float = 0.0  # 周向移動
    f_lean: float = 0.0  # 傾斜移動（垂直於弦）
    f_sweep: float = 0.0  # 掃蕩移動（沿弦線）
    f_axial: float = 0.0  # 軸向移動

    # 縮放參數
    f_scale: float = 1.0  # 弦長縮放因子
    f_const: float = 0.0  # 固定點分數位置（0=前緣，1=後緣）

    # HUB 截面質心（用於堆疊參考）
    x_centroid_hub: float = 0.0
    y_centroid_hub: float = 0.0


@dataclass
class BladeSection3D:
    """3D 葉片截面（投影到流線表面後）。"""

    section_number: int  # 截面號（1=HUB，NOSECT=TIP）
    spanwise_fraction: float  # 跨向分數位置 (0=HUB, 1=TIP)

    # 網格點坐標（投影後）
    x_grid: list[float] = field(default_factory=list)  # 軸向坐標 [m]
    y_grid: list[float] = field(default_factory=list)  # 周向坐標 [m]
    r_grid: list[float] = field(default_factory=list)  # 半徑坐標 [m]
    s_grid: list[float] = field(default_factory=list)  # 子午線距離 [m]
    tk_grid: list[float] = field(default_factory=list)  # 厚度 [m]

    # 質心（用於堆疊）
    x_centroid: float = 0.0
    y_centroid: float = 0.0

    # 前後緣位置索引
    j_le: int = 0  # 前緣點索引
    j_te: int = 0  # 後緣點索引


@dataclass
class ThicknessParameters:
    """厚度分佈參數（INTYPE=1）。"""

    tk_le: float = 0.02  # 前緣厚度比（相對於弦長）
    tk_te: float = 0.01  # 後緣厚度比
    tk_max: float = 0.10  # 最大厚度比
    xtk_max: float = 0.40  # 最大厚度位置（軸向分數）

    # 厚度分佈類型參數
    tk_type: float = 2.0  # 厚度分佈指數（控制厚度曲線形狀）
    le_exp: float = 3.0  # 前緣橢圓化指數
    xmod_le: float = 0.02  # 前緣修正範圍（軸向分數）
    xmod_te: float = 0.01  # 後緣修正範圍（軸向分數）

    # 厚度方向控制
    f_perp: float = 1.0  # 垂直於中弧線的厚度比例 (0-1)


@dataclass
class GridParameters:
    """網格參數。"""

    # 網格尺寸
    im: int = 37  # 周向網格點數
    km: int = 11  # 跨向網格點數

    # 周向網格擴張
    fp_rat: float = 1.25  # 周向擴張比
    fp_max: float = 20.0  # 周向最大擴張比

    # 跨向網格擴張
    fr_rat: float = 1.25  # 跨向擴張比
    fr_max: float = 20.0  # 跨向最大擴張比

    # 軸向網格點數
    nint_up: int = 5  # 上游點數
    nint_on: int = 50  # 葉片上點數
    nint_dn: int = 10  # 下游點數


@dataclass
class BladeRow:
    """葉片排（行）數據。"""

    row_number: int  # 排號
    row_type: str  # 'R' = 轉子, 'S' = 定子
    n_blade: int  # 葉片數
    rpm: float = 0.0  # 轉速 [RPM]（定子為0）

    # 截面列表
    sections: list[BladeSection3D] = field(default_factory=list)

    # 網格參數
    grid_params: GridParameters | None = None

    # 前後緣索引（整個行）
    j_le: int = 0
    j_te: int = 0
    j_m: int = 0  # 總網格點數


@dataclass
class StagenConfig:
    """STAGEN 配置參數。"""

    # 氣體性質
    rgas: float  # 氣體常數 [J/(kg·K)]
    gamma: float  # 比熱比

    # 行數據
    nrows: int  # 行數（葉片排數）
    nosect: int  # 每行的截面數

    # 網格參數
    grid_params: GridParameters

    # 坐標縮放因子
    fac_scale: float = 1.0

    # 葉片排列表
    blade_rows: list[BladeRow] = field(default_factory=list)


# 常數定義
STAGEN_CONSTANTS = {
    "PI": 3.141592653589793,
    "DEG2RAD": 0.017453292519943295,
    "RAD2DEG": 57.29577951308232,
    "NSURF_MAX": 1000,  # 最大表面點數
    "NSPAN_MAX": 100,  # 最大跨向分層
    "NXON_MAX": 500,  # 最大軸向網格點數
    "NPIT_MAX": 100,  # 最大周向網格點數
    "NSS_MAX": 100,  # 最大流線表面點數
    "NSTAGE_MAX": 25,  # 最大級數
}
