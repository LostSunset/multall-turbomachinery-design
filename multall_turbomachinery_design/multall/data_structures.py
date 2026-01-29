# -*- coding: utf-8 -*-
"""MULTALL 數據結構定義。

定義 3D Navier-Stokes 求解器所需的核心數據結構。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np
from numpy.typing import NDArray


class TimeStepType(IntEnum):
    """時間步進類型。"""

    SCREE = 3  # SCREE 格式（預設）
    USER_DEFINED = 4  # 用戶定義係數
    ARTIFICIAL_COMPRESSIBILITY = 5  # 人工可壓縮性
    INCOMPRESSIBLE = 6  # 不可壓縮流


class ViscousModel(IntEnum):
    """黏性模型類型。"""

    INVISCID = 0  # 無黏
    MIXING_LENGTH = 100  # 混合長度模型
    NEWLOS = 150  # 新混合長度模型（完整 N-S）
    SPALART_ALLMARAS = 200  # Spalart-Allmaras 湍流模型


class GasType(IntEnum):
    """氣體類型。"""

    PERFECT_GAS = 0  # 完美氣體
    VARIABLE_CP = 1  # CP 隨溫度變化
    LOOKUP_TABLE = 3  # 查表法（蒸汽等）


class BoundaryConditionType(Enum):
    """邊界條件類型。"""

    TOTAL_PRESSURE_TEMPERATURE = "PT"  # 總壓總溫
    MASS_FLOW = "MF"  # 質量流量
    STATIC_PRESSURE = "PS"  # 靜壓
    PERIODIC = "PER"  # 週期性


@dataclass
class GasProperties:
    """氣體性質。"""

    # 基本性質
    cp: float = 1005.0  # 定壓比熱 [J/(kg·K)]
    gamma: float = 1.4  # 比熱比
    rgas: float = 287.0  # 氣體常數 [J/(kg·K)]

    # 氣體類型
    gas_type: GasType = GasType.PERFECT_GAS

    # 變 CP 氣體係數 (CP = CP1 + CP2*T + CP3*T^2)
    cp1: float = 1272.5
    cp2: float = 0.2125
    cp3: float = 0.000015625
    tref: float = 1400.0  # 參考溫度 [K]

    # Prandtl 數
    prandtl: float = 0.72

    def __post_init__(self) -> None:
        """計算派生性質。"""
        if self.gas_type == GasType.PERFECT_GAS:
            self.rgas = self.cp * (self.gamma - 1.0) / self.gamma
        self.cv = self.cp / self.gamma
        self.fga = (self.gamma + 1.0) / (2.0 * self.gamma)
        self.ga1 = self.gamma - 1.0
        self.rcp = 1.0 / self.cp
        self.rcv = 1.0 / self.cv


@dataclass
class GridParameters:
    """網格參數。"""

    # 網格尺寸
    im: int = 37  # 周向網格點數
    jm: int = 100  # 軸向網格點數（每行）
    km: int = 11  # 跨向網格點數

    # 相對間距（周向）
    fp: list[float] = field(default_factory=list)

    # 相對間距（跨向）
    fr: list[float] = field(default_factory=list)

    # 多重網格參數
    ir: int = 3  # I 方向塊大小
    jr: int = 3  # J 方向塊大小
    kr: int = 3  # K 方向塊大小
    irbb: int = 9  # I 方向大塊大小
    jrbb: int = 9  # J 方向大塊大小
    krbb: int = 9  # K 方向大塊大小

    # 多重網格時間步因子
    fblk1: float = 0.4
    fblk2: float = 0.2
    fblk3: float = 0.1


@dataclass
class SolverParameters:
    """求解器參數。"""

    # 時間步進
    time_step_type: TimeStepType = TimeStepType.SCREE
    cfl: float = 0.4  # CFL 數
    damping: float = 10.0  # 阻尼因子
    mach_limit: float = 2.0  # 馬赫數限制

    # 收斂控制
    max_steps: int = 5000  # 最大時間步數
    convergence_limit: float = 0.005  # 收斂準則

    # 平滑因子
    sfx_in: float = 0.005  # X 方向平滑
    sft_in: float = 0.005  # T 方向平滑
    fac_4th: float = 0.8  # 四階平滑比例
    nchange: int = 1250  # 平滑減少步數

    # SSS 格式係數
    f1: float = 2.0
    f2: float = -1.0
    f3: float = 0.0
    f2eff: float = -1.0
    nrsmth: int = 0
    rsmth: float = 0.40

    # 人工可壓縮性參數（ITIMST=5,6）
    vsound: float = 150.0  # 人工聲速
    rf_ptru: float = 0.01  # 密度鬆弛因子
    rf_vsound: float = 0.002  # 聲速鬆弛因子
    vs_vmax: float = 2.0  # 聲速/最大速度比
    density: float = 1.20  # 不可壓縮密度

    # 重啟選項
    restart: bool = False
    inverse_design: bool = False


@dataclass
class ViscousParameters:
    """黏性模型參數。"""

    # 模型類型
    model: ViscousModel = ViscousModel.MIXING_LENGTH
    nlos: int = 5  # 每 NLOS 步更新黏性力

    # 基本參數
    reynolds: float = 500000.0  # 雷諾數
    rf_vis: float = 0.5  # 黏性項鬆弛因子
    ftrans: float = 0.0001  # 轉捩因子
    turbvis_limit: float = 3000.0  # 湍流/層流黏度比限制

    # 壁面 y+ 範圍
    yplus_lam: float = 5.0  # 層流 y+
    yplus_turb: float = 25.0  # 湍流 y+

    # Spalart-Allmaras 模型係數
    fac_stmix: float = 0.0
    fac_st0: float = 1.0
    fac_st1: float = 1.0
    fac_st2: float = 1.0
    fac_st3: float = 1.0
    fac_sfvis: float = 2.0
    fac_vort: float = 0.0
    fac_pgrad: float = 0.0


@dataclass
class MixingPlaneParameters:
    """混合平面參數。"""

    enabled: bool = True
    rfmix: float = 0.025  # 混合平面鬆弛因子
    fsmthb: float = 1.0  # 平滑因子
    fextrap: float = 0.80  # 外推因子
    fangle: float = 0.80  # 角度因子


@dataclass
class InletBoundary:
    """進口邊界條件。"""

    # 邊界類型選項
    use_total_pressure: bool = True  # 使用總壓
    use_tangential_velocity: bool = False  # 指定切向速度
    use_radial_velocity: bool = True  # 指定徑向速度
    use_mass_flow: bool = False  # 指定質量流量
    repeating_stage: bool = False  # 重複級

    # 鬆弛因子
    rfin: float = 0.1

    # 分佈數據（從 HUB 到 TIP）
    po: list[float] = field(default_factory=list)  # 總壓 [Pa]
    to: list[float] = field(default_factory=list)  # 總溫 [K]
    alpha: list[float] = field(default_factory=list)  # 流動角 [deg]
    beta: list[float] = field(default_factory=list)  # 俯仰角 [deg]
    vtan: list[float] = field(default_factory=list)  # 切向速度 [m/s]

    # 質量流量
    mass_flow: float = 0.0  # [kg/s]
    rflow: float = 0.1  # 質量流量鬆弛因子


@dataclass
class ExitBoundary:
    """出口邊界條件。"""

    # 邊界類型
    use_static_pressure: bool = True

    # 靜壓分佈
    pstatic_hub: float = 101325.0  # HUB 靜壓 [Pa]
    pstatic_tip: float = 101325.0  # TIP 靜壓 [Pa]

    # 平滑和外推
    sfexit: float = 0.0  # 出口平滑因子
    nsfexit: int = 0  # 平滑次數
    fp_xtrap: float = 1.0  # 外推因子
    fracwave: float = 0.0  # 波傳播因子

    # 節流閥選項
    plate_loss: float = 0.0  # 孔板損失係數
    throttle_enabled: bool = False
    throttle_pressure: float = 0.0
    throttle_mass: float = 0.0
    rfthrottle: float = 0.1


@dataclass
class BladeRowGeometry:
    """葉片排幾何。"""

    row_number: int  # 排號
    row_type: str  # 'R' = 轉子, 'S' = 定子
    n_blades: int  # 葉片數
    rpm: float = 0.0  # 轉速 [RPM]

    # 葉片排在機械中的位置
    stage_number: int = 1

    # 網格點索引
    j_start: int = 0  # 起始 J 索引
    j_end: int = 0  # 結束 J 索引
    j_le: list[int] = field(default_factory=list)  # 前緣 J 索引（每個 K）
    j_te: list[int] = field(default_factory=list)  # 後緣 J 索引（每個 K）
    j_mix: int = 0  # 混合平面 J 索引

    # 幾何數據（每個 K 截面）
    chord: float = 0.0  # 弦長 [m]
    pitch: float = 0.0  # 節距 [m]

    # 葉尖間隙
    tip_gap_enabled: bool = False
    tip_gap_fraction: float = 0.0  # 相對間隙

    # 護罩洩漏
    shroud_enabled: bool = False

    # 表面粗糙度
    roughness_hub: float = 0.0
    roughness_tip: float = 0.0

    # 逆向設計選項
    redesign_enabled: bool = False


@dataclass
class BladeSection:
    """葉片截面幾何。"""

    k_index: int  # 跨向索引

    # 坐標數據
    x: list[float] = field(default_factory=list)  # 軸向坐標 [m]
    r: list[float] = field(default_factory=list)  # 半徑坐標 [m]
    rt_upper: list[float] = field(default_factory=list)  # 吸力面 R*θ [m]
    rt_thickness: list[float] = field(default_factory=list)  # 厚度 [m]

    # 流道數據
    x_hub: list[float] = field(default_factory=list)
    r_hub: list[float] = field(default_factory=list)
    x_tip: list[float] = field(default_factory=list)
    r_tip: list[float] = field(default_factory=list)


@dataclass
class FlowField:
    """流場數據。"""

    # 網格尺寸
    im: int
    jm: int
    km: int

    # 原始變量 (primitive variables)
    rho: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 密度
    p: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 壓力
    vx: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 軸向速度
    vr: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 徑向速度
    vt: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 切向速度
    t_static: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 靜溫

    # 守恆變量 (conservative variables)
    ro: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # ρ
    rovx: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # ρVx
    rovr: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # ρVr
    rorvt: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # ρrVθ
    roe: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # ρE

    # 總條件
    ho: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 總焓
    po: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 總壓
    to: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 總溫

    # 馬赫數
    mach: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 絕對馬赫數
    mach_rel: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 相對馬赫數

    def initialize(self) -> None:
        """初始化數組。"""
        shape = (self.im, self.jm, self.km)
        self.rho = np.zeros(shape)
        self.p = np.zeros(shape)
        self.vx = np.zeros(shape)
        self.vr = np.zeros(shape)
        self.vt = np.zeros(shape)
        self.t_static = np.zeros(shape)

        self.ro = np.zeros(shape)
        self.rovx = np.zeros(shape)
        self.rovr = np.zeros(shape)
        self.rorvt = np.zeros(shape)
        self.roe = np.zeros(shape)

        self.ho = np.zeros(shape)
        self.po = np.zeros(shape)
        self.to = np.zeros(shape)

        self.mach = np.zeros(shape)
        self.mach_rel = np.zeros(shape)


@dataclass
class Grid3D:
    """3D 網格數據。"""

    # 網格尺寸
    im: int
    jm: int
    km: int

    # 坐標
    x: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 軸向
    r: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 半徑
    theta: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # 周向角
    rtheta: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # R*θ

    # 子午線距離
    s_merid: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 單元體積
    vol: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # 面積向量
    abx: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # B 面 X 分量
    abr: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # B 面 R 分量
    abt: NDArray[np.float64] = field(default_factory=lambda: np.array([]))  # B 面 θ 分量

    def initialize(self) -> None:
        """初始化數組。"""
        shape_2d = (self.jm, self.km)
        shape_3d = (self.im, self.jm, self.km)

        self.x = np.zeros(shape_2d)
        self.r = np.zeros(shape_2d)
        self.s_merid = np.zeros(shape_2d)

        self.theta = np.zeros(shape_3d)
        self.rtheta = np.zeros(shape_3d)
        self.vol = np.zeros(shape_3d)

        self.abx = np.zeros(shape_3d)
        self.abr = np.zeros(shape_3d)
        self.abt = np.zeros(shape_2d)


@dataclass
class MultallConfig:
    """MULTALL 配置。"""

    # 標題
    title: str = "MULTALL Calculation"

    # 氣體性質
    gas: GasProperties = field(default_factory=GasProperties)

    # 網格參數
    grid: GridParameters = field(default_factory=GridParameters)

    # 求解器參數
    solver: SolverParameters = field(default_factory=SolverParameters)

    # 黏性模型
    viscous: ViscousParameters = field(default_factory=ViscousParameters)

    # 混合平面
    mixing_plane: MixingPlaneParameters = field(default_factory=MixingPlaneParameters)

    # 邊界條件
    inlet: InletBoundary = field(default_factory=InletBoundary)
    exit: ExitBoundary = field(default_factory=ExitBoundary)

    # 葉片排列表
    nrows: int = 0
    blade_rows: list[BladeRowGeometry] = field(default_factory=list)

    # 冷卻和放氣
    cooling_enabled: bool = False
    bleed_enabled: bool = False
    roughness_enabled: bool = False

    # 截面數
    nsections: int = 5


# 常數定義
MULTALL_CONSTANTS = {
    "PI": 3.141592653589793,
    "DEG2RAD": 0.017453292519943295,
    "RAD2DEG": 57.29577951308232,
    # 最大數組維度（參考 FORTRAN）
    "ID_MAX": 64,  # 最大周向網格點
    "JD_MAX": 2500,  # 最大軸向網格點
    "KD_MAX": 82,  # 最大跨向網格點
    "NRS_MAX": 21,  # 最大葉片排數
}
