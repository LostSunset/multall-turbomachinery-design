# -*- coding: utf-8 -*-
"""MEANGEN 數據結構定義。

定義 MEANGEN 模組使用的主要數據類別和結構。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MachineType(Enum):
    """機械類型。"""

    TURBINE = "T"  # 渦輪
    COMPRESSOR = "C"  # 壓縮機


class FlowType(Enum):
    """流動類型。"""

    AXIAL = "AXI"  # 軸向流
    MIXED = "MIX"  # 混流
    RADIAL = "RAD"  # 徑向流


class InputType(Enum):
    """速度三角形輸入類型。"""

    TYPE_A = "A"  # 反應度/流量係數/負荷係數法
    TYPE_B = "B"  # 流量/定子角/轉子角法
    TYPE_C = "C"  # 其他組合
    TYPE_D = "D"  # 其他組合


class RadiusType(Enum):
    """半徑輸入類型。"""

    DIRECT = "A"  # 直接輸入設計半徑
    ENTHALPY = "B"  # 指定焓變


@dataclass
class GasProperties:
    """氣體性質。"""

    rgas: float  # 氣體常數 [J/(kg·K)]
    gamma: float  # 比熱比
    poin: float  # 進口總壓 [Bar]
    toin: float  # 進口總溫 [K]


@dataclass
class VelocityTriangle:
    """速度三角形。"""

    vm: float  # 子午速度 [m/s]
    vtheta: float  # 切向速度 [m/s]
    u: float  # 圓周速度 [m/s]
    alpha: float  # 絕對流角 [度]
    beta: float  # 相對流角 [度]
    mach_abs: float  # 絕對馬赫數
    mach_rel: float  # 相對馬赫數


@dataclass
class BladeRow:
    """葉片排。"""

    row_number: int  # 排號
    row_type: str  # 'R' = 轉子, 'S' = 定子
    n_blades: int  # 葉片數
    rpm: float  # 轉速 [RPM]
    axial_chord: float  # 軸向弦長 [m]

    # 進出口角度
    alpha_in: float  # 進口絕對角 [度]
    alpha_out: float  # 出口絕對角 [度]
    beta_in: float  # 進口相對角 [度]
    beta_out: float  # 出口相對角 [度]

    # 厚度參數
    tk_max: float = 0.0  # 最大厚度比
    tk_le: float = 0.0  # 前緣厚度比
    tk_te: float = 0.0  # 後緣厚度比
    xtk_max: float = 0.45  # 最大厚度位置

    # 偏差和入射角
    incidence: float = 0.0  # 入射角 [度]
    deviation: float = 0.0  # 偏角 [度]


@dataclass
class StreamSurface:
    """流表面。"""

    npoints: int  # 點數
    x: list[float] = field(default_factory=list)  # 軸向坐標 [m]
    r: list[float] = field(default_factory=list)  # 徑向坐標 [m]
    vm_ratio: list[float] = field(default_factory=list)  # 子午速度比
    pitch_angle: list[float] = field(default_factory=list)  # 俯仰角 [度]

    # 流表面距離
    s_dist: list[float] = field(default_factory=list)  # 沿流表面距離 [m]

    # 堵塞因子 (v17.4)
    fblock: list[float] = field(default_factory=list)  # 堵塞因子


@dataclass
class StageDesign:
    """級設計參數。"""

    stage_number: int  # 級號

    # 速度係數
    phi: float  # 流量係數
    psi: float  # 負荷係數
    reaction: float  # 反應度

    # 設計半徑
    r_design: float  # 設計半徑 [m]
    r_hub: float = 0.0  # 轂部半徑 [m]
    r_tip: float = 0.0  # 葉尖半徑 [m]

    # 效率和損失
    efficiency: float = 0.9  # 等熵效率
    dho: float = 0.0  # 焓變 [J/kg]
    dho_is: float = 0.0  # 等熵焓變 [J/kg]

    # 輸入類型和角度
    input_type: InputType = InputType.TYPE_A  # 輸入類型
    alpha_in: float = 0.0  # 進口絕對角 [度]（Type B）
    alpha_out: float = 0.0  # 出口絕對角 [度]（Type B）
    beta_in: float = 0.0  # 進口相對角 [度]（Type B）
    beta_out: float = 0.0  # 出口相對角 [度]（Type B）

    # 幾何參數
    axial_chord_1: float | None = None  # 轉子軸向弦長 [m]
    axial_chord_2: float | None = None  # 定子軸向弦長 [m]
    row_gap: float | None = None  # 行間隙 [m]
    stage_gap: float | None = None  # 級間隙 [m]

    # 偏差和入射角
    ainc1: float | None = None  # 轉子入射角 [度]
    ainc2: float | None = None  # 定子入射角 [度]
    devn1: float | None = None  # 轉子偏角 [度]
    devn2: float | None = None  # 定子偏角 [度]

    # 性能輸出
    work_output: float = 0.0  # 比功 [J/kg]
    loading_coefficient: float = 0.0  # 負荷係數
    inlet_triangle: VelocityTriangle | None = None  # 進口速度三角形
    outlet_triangle: VelocityTriangle | None = None  # 出口速度三角形

    # 葉片排
    rotor: BladeRow | None = None  # 轉子
    stator: BladeRow | None = None  # 定子

    # 流表面
    hub_surface: StreamSurface | None = None  # 轂部流表面
    tip_surface: StreamSurface | None = None  # 葉尖流表面
    mean_surface: StreamSurface | None = None  # 平均流表面
    stream_surface: StreamSurface | None = None  # 設計點流表面

    # 堵塞因子 (v17.4)
    fblock_le: float = 0.0  # 前緣堵塞因子
    fblock_te: float = 0.02  # 後緣堵塞因子

    # 扭轉控制 (v17.4)
    frac_twist: float = 1.0  # 扭轉比例 (0=無扭轉, 1=完全自由渦)

    # 熱力學狀態（由求解器計算）
    thermodynamics: StageThermodynamics | None = None


@dataclass
class MeangenConfig:
    """MEANGEN 配置參數。"""

    # 機械類型
    machine_type: MachineType
    flow_type: FlowType

    # 氣體性質
    gas: GasProperties

    # 運行參數
    nstages: int  # 級數
    rpm: float  # 轉速 [RPM]
    mass_flow: float  # 質量流量 [kg/s]

    # 設計參數
    design_radius: float  # 設計半徑 [m]
    design_location: str = "M"  # 'H'=轂部, 'M'=中間, 'T'=葉尖

    # 級設計
    stages: list[StageDesign] = field(default_factory=list)

    # 輸出選項
    nosect: int = 3  # 葉片截面數 (通常為3: 葉根、中間、葉尖)
    im: int = 37  # 網格點數
    km: int = 11  # 網格點數

    # 默認參數
    zweifel_stator: float = 0.85  # Zweifel係數 (定子)
    zweifel_rotor: float = 0.85  # Zweifel係數 (轉子)


@dataclass
class FlowState:
    """流動狀態。"""

    # 熱力性質
    ho: float  # 總焓 [J/kg]
    h: float = 0.0  # 靜焓 [J/kg]
    po: float = 0.0  # 總壓 [Pa]
    p: float = 0.0  # 靜壓 [Pa]
    to: float = 0.0  # 總溫 [K]
    t: float = 0.0  # 靜溫 [K]
    s: float = 0.0  # 熵 [J/(kg·K)]
    rho: float = 0.0  # 密度 [kg/m³]

    # 速度
    v: float = 0.0  # 速度 [m/s]
    vs: float = 0.0  # 聲速 [m/s]
    mach: float = 0.0  # 馬赫數

    # 濕度（蒸汽）
    wetness: float = 0.0  # 濕度分數


@dataclass
class StageThermodynamics:
    """級熱力學狀態。"""

    # 進口狀態
    inlet_state: FlowState
    # 出口狀態
    outlet_state: FlowState
    # 等熵出口狀態（理想狀態）
    outlet_isentropic: FlowState

    # 效率和性能
    isentropic_efficiency: float = 0.0
    polytropic_efficiency: float = 0.0

    # 壓比和溫比
    pressure_ratio: float = 1.0
    temperature_ratio: float = 1.0

    # 壅塞檢查
    is_choked: bool = False
    critical_mach: float = 1.0


# 常數定義
CONSTANTS = {
    "DEG2RAD": 0.017453292519943295,  # π/180
    "RAD2DEG": 57.29577951308232,  # 180/π
    "PI": 3.141592653589793,
    "GRAVITY": 9.80665,  # 重力加速度 [m/s²]
}
