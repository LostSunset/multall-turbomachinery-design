# -*- coding: utf-8 -*-
"""MULTALL 主求解器。

3D Navier-Stokes 求解器，用於渦輪機械流場計算。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .data_structures import (
    FlowField,
    Grid3D,
    MultallConfig,
    ViscousModel,
)
from .flux import ArtificialViscosity, FluxCalculator
from .gas_properties import GasCalculator
from .io_handler import MultallFileHandler, MultallInputReader
from .time_stepping import ConvergenceMonitor, TimeStepper

if TYPE_CHECKING:
    from collections.abc import Callable


class MultallSolver:
    """MULTALL 3D 流場求解器。

    使用時間推進法求解穩態 3D 流場。
    支援多級渦輪機械、混合平面模型和多種黏性模型。
    """

    def __init__(self, config: MultallConfig) -> None:
        """初始化求解器。

        Args:
            config: MULTALL 配置
        """
        self.config = config
        self.gas_calc = GasCalculator(config.gas)

        # 初始化數值組件
        self.flux_calc = FluxCalculator(config.gas)
        self.time_stepper = TimeStepper(config.gas, config.solver)
        self.artificial_viscosity = ArtificialViscosity(
            sf_2nd=config.solver.sfx_in,
            sf_4th=config.solver.fac_4th,
        )
        self.convergence_monitor = ConvergenceMonitor(
            convergence_limit=config.solver.convergence_limit,
        )

        # 初始化網格和流場
        self.grid: Grid3D | None = None
        self.flow: FlowField | None = None

        # 求解狀態
        self._step = 0
        self._converged = False

        # 回調函數
        self._progress_callback: Callable[[int, float, float], None] | None = None

    @classmethod
    def from_input_file(cls, input_file: str | Path) -> MultallSolver:
        """從輸入文件創建求解器。

        Args:
            input_file: 輸入文件路徑

        Returns:
            求解器實例
        """
        reader = MultallInputReader()
        config = reader.read(input_file)
        return cls(config)

    def set_progress_callback(self, callback: Callable[[int, float, float], None]) -> None:
        """設置進度回調函數。

        Args:
            callback: 回調函數 (step, residual, mass_flow)
        """
        self._progress_callback = callback

    def initialize_grid(self) -> None:
        """初始化計算網格。"""
        cfg = self.config.grid
        # 計算總軸向點數（所有葉片排）
        jm_total = cfg.jm * self.config.nrows if self.config.nrows > 0 else cfg.jm

        self.grid = Grid3D(im=cfg.im, jm=jm_total, km=cfg.km)
        self.grid.initialize()

    def initialize_flow(self) -> None:
        """初始化流場。"""
        if self.grid is None:
            self.initialize_grid()

        assert self.grid is not None
        self.flow = FlowField(im=self.grid.im, jm=self.grid.jm, km=self.grid.km)
        self.flow.initialize()

        # 設置初始條件
        self._set_initial_conditions()

    def _set_initial_conditions(self) -> None:
        """設置初始流場條件。"""
        if self.flow is None:
            return

        inlet = self.config.inlet
        gas = self.gas_calc

        # 使用進口條件初始化
        if inlet.po and inlet.to:
            # 使用進口總壓總溫
            po_avg = np.mean(inlet.po)
            to_avg = np.mean(inlet.to)

            # 假設初始馬赫數 0.3
            mach_init = 0.3
            p_static, t_static = gas.static_from_total(po_avg, to_avg, mach_init)
            rho = gas.density(p_static, t_static)
            a = gas.speed_of_sound(t_static)
            v_init = mach_init * a

            # 填充整個域
            self.flow.rho[:] = rho
            self.flow.p[:] = p_static
            self.flow.t_static[:] = t_static
            self.flow.vx[:] = v_init
            self.flow.vr[:] = 0.0
            self.flow.vt[:] = 0.0

            # 計算守恆變量
            self._update_conservative_variables()
        else:
            # 使用默認條件
            self.flow.rho[:] = 1.2
            self.flow.p[:] = 101325.0
            self.flow.t_static[:] = 288.15
            self.flow.vx[:] = 100.0
            self.flow.vr[:] = 0.0
            self.flow.vt[:] = 0.0

            self._update_conservative_variables()

    def _update_conservative_variables(self) -> None:
        """從原始變量更新守恆變量。"""
        if self.flow is None or self.grid is None:
            return

        flow = self.flow

        # ρ
        flow.ro[:] = flow.rho

        # ρVx
        flow.rovx[:] = flow.rho * flow.vx

        # ρVr
        flow.rovr[:] = flow.rho * flow.vr

        # ρrVθ（需要半徑）
        # 這裡簡化處理，假設 r 在 J-K 平面上定義
        # flow.rorvt[:] = flow.rho * r * flow.vt

        # ρE（總能量）
        e_internal = self.gas_calc.internal_energy(flow.t_static)
        v_sq = flow.vx**2 + flow.vr**2 + flow.vt**2
        flow.roe[:] = flow.rho * (e_internal + 0.5 * v_sq)

        # 總焓
        flow.ho[:] = self.gas_calc.total_enthalpy(flow.t_static, flow.vx, flow.vr, flow.vt)

    def _update_primitive_variables(self) -> None:
        """從守恆變量更新原始變量。"""
        if self.flow is None:
            return

        flow = self.flow
        gas = self.gas_calc

        # 密度
        flow.rho[:] = flow.ro

        # 速度
        rho_safe = np.maximum(flow.rho, 1e-10)
        flow.vx[:] = flow.rovx / rho_safe
        flow.vr[:] = flow.rovr / rho_safe
        # flow.vt[:] = flow.rorvt / (rho_safe * r)

        # 內能和溫度
        v_sq = flow.vx**2 + flow.vr**2 + flow.vt**2
        e_total = flow.roe / rho_safe
        e_internal = e_total - 0.5 * v_sq

        # T = e / cv
        flow.t_static[:] = np.maximum(e_internal / gas.cv, 100.0)

        # 壓力
        flow.p[:] = gas.pressure(flow.rho, flow.t_static)

    def _compute_fluxes(
        self,
    ) -> tuple[
        tuple[NDArray[np.float64], ...],
        tuple[NDArray[np.float64], ...],
        tuple[NDArray[np.float64], ...],
    ]:
        """計算三個方向的通量。

        Returns:
            (X方向通量, θ方向通量, R方向通量)
            每個方向包含 (質量通量, X動量通量, R動量通量, θ動量通量, 能量通量)
        """
        if self.flow is None or self.grid is None:
            empty = (np.array([]),) * 5
            return empty, empty, empty

        # 計算三個方向的對流通量
        flux_x = self.flux_calc.compute_convective_flux_x(self.flow, self.grid)
        flux_theta = self.flux_calc.compute_convective_flux_theta(self.flow, self.grid)
        flux_r = self.flux_calc.compute_convective_flux_r(self.flow, self.grid)

        return flux_x, flux_theta, flux_r

    def _compute_residual_from_fluxes(
        self,
        flux_x: tuple[NDArray[np.float64], ...],
        flux_theta: tuple[NDArray[np.float64], ...],
        flux_r: tuple[NDArray[np.float64], ...],
    ) -> tuple[NDArray[np.float64], ...]:
        """從通量計算殘差。

        Args:
            flux_x: X 方向通量
            flux_theta: θ 方向通量
            flux_r: R 方向通量

        Returns:
            各守恆量的殘差
        """
        if self.flow is None or self.grid is None:
            return (np.array([]),) * 5

        return self.flux_calc.compute_residual(self.flow, self.grid, flux_x, flux_theta, flux_r)

    def _apply_boundary_conditions(self) -> None:
        """應用邊界條件。"""
        if self.flow is None:
            return

        # 進口邊界
        self._apply_inlet_bc()

        # 出口邊界
        self._apply_exit_bc()

        # 周期性邊界
        self._apply_periodic_bc()

        # 壁面邊界
        self._apply_wall_bc()

    def _apply_inlet_bc(self) -> None:
        """應用進口邊界條件。"""
        if self.flow is None:
            return

        inlet = self.config.inlet
        gas = self.gas_calc

        if inlet.use_total_pressure and inlet.po and inlet.to:
            # 從總壓總溫設置進口
            j = 0  # 進口面
            for k in range(self.flow.km):
                # 插值獲取該截面的總條件
                po = inlet.po[min(k, len(inlet.po) - 1)]
                to = inlet.to[min(k, len(inlet.to) - 1)]

                # 從內部外推馬赫數
                mach = float(np.mean(self.flow.mach[:, j + 1, k]))
                mach = max(0.01, min(mach, 0.99))

                # 計算靜態條件
                p_static, t_static = gas.static_from_total(po, to, mach)
                rho = gas.density(p_static, t_static)

                # 設置進口值
                self.flow.rho[:, j, k] = rho
                self.flow.p[:, j, k] = p_static
                self.flow.t_static[:, j, k] = t_static

    def _apply_exit_bc(self) -> None:
        """應用出口邊界條件。"""
        if self.flow is None:
            return

        exit_bc = self.config.exit

        if exit_bc.use_static_pressure:
            j = self.flow.jm - 1  # 出口面
            for k in range(self.flow.km):
                # 線性插值 HUB 到 TIP 靜壓
                frac = k / max(self.flow.km - 1, 1)
                p_exit = exit_bc.pstatic_hub * (1 - frac) + exit_bc.pstatic_tip * frac

                # 從內部外推其他變量
                self.flow.p[:, j, k] = p_exit

    def _apply_periodic_bc(self) -> None:
        """應用周期性邊界條件。"""
        if self.flow is None:
            return

        # I=1 和 I=IM 面周期性
        self.flow.rho[0, :, :] = self.flow.rho[-2, :, :]
        self.flow.rho[-1, :, :] = self.flow.rho[1, :, :]

        self.flow.vx[0, :, :] = self.flow.vx[-2, :, :]
        self.flow.vx[-1, :, :] = self.flow.vx[1, :, :]

        self.flow.vr[0, :, :] = self.flow.vr[-2, :, :]
        self.flow.vr[-1, :, :] = self.flow.vr[1, :, :]

        self.flow.vt[0, :, :] = self.flow.vt[-2, :, :]
        self.flow.vt[-1, :, :] = self.flow.vt[1, :, :]

        self.flow.p[0, :, :] = self.flow.p[-2, :, :]
        self.flow.p[-1, :, :] = self.flow.p[1, :, :]

    def _apply_wall_bc(self) -> None:
        """應用壁面邊界條件（HUB 和 CASING）。"""
        if self.flow is None:
            return

        # K=1 (HUB) - 無滑移壁面
        # 法向速度為零
        self.flow.vr[:, :, 0] = 0.0

        # K=KM (CASING) - 無滑移壁面
        self.flow.vr[:, :, -1] = 0.0

    def _compute_time_step(self) -> NDArray[np.float64]:
        """計算局部時間步長。

        Returns:
            時間步長數組
        """
        if self.flow is None or self.grid is None:
            return np.array([])

        cfg = self.config.solver
        gas = self.gas_calc

        # 計算聲速
        a = gas.speed_of_sound(self.flow.t_static)

        # 計算特徵速度
        v_mag = np.sqrt(self.flow.vx**2 + self.flow.vr**2 + self.flow.vt**2)
        lambda_max = v_mag + a

        # 估計網格間距（簡化）
        dx = 0.001  # 需要從網格計算

        # CFL 條件
        dt = cfg.cfl * dx / np.maximum(lambda_max, 1e-10)

        return dt

    def _time_step(self) -> float:
        """執行一個時間步。

        Returns:
            殘差 L2 範數
        """
        if self.flow is None or self.grid is None:
            return 1.0

        # 計算三個方向的通量
        flux_x, flux_theta, flux_r = self._compute_fluxes()

        # 計算殘差（通量散度）
        residual = self._compute_residual_from_fluxes(flux_x, flux_theta, flux_r)

        # 計算局部時間步長
        dt = self.time_stepper.compute_local_time_step(self.flow, self.grid)

        # 添加人工黏性（穩定性）
        av_x = self.artificial_viscosity.compute_artificial_dissipation(self.flow, direction="x")

        # 合併人工黏性到殘差
        combined_residual = (
            residual[0] - av_x[0],
            residual[1] - av_x[1],
            residual[2] - av_x[2],
            residual[3] - av_x[3],
            residual[4] - av_x[4],
        )

        # 執行時間步進（使用 SCREE 或 Euler 方法）
        self.time_stepper.scree_step(self.flow, combined_residual, dt)

        # 更新原始變量
        self._update_primitive_variables()

        # 應用邊界條件
        self._apply_boundary_conditions()

        # 計算 L2 殘差並記錄
        l2_residual = self.convergence_monitor.compute_l2_residual(residual)
        self.convergence_monitor.add_residual(l2_residual)

        return l2_residual

    def _compute_residual(self) -> float:
        """計算殘差。

        Returns:
            均方根殘差
        """
        if self.flow is None:
            return 1.0

        # 使用質量守恆殘差
        # 這裡簡化為密度變化
        residual = float(np.std(self.flow.rho))
        return residual

    def _compute_mass_flow(self) -> float:
        """計算質量流量。

        Returns:
            質量流量 [kg/s]
        """
        if self.flow is None or self.grid is None:
            return 0.0

        # 在進口面計算質量流量
        j = 1
        mass_flow = float(np.sum(self.flow.rho[:, j, :] * self.flow.vx[:, j, :]))
        # 需要乘以面積

        return mass_flow

    def solve(self, max_steps: int | None = None) -> bool:
        """執行求解。

        Args:
            max_steps: 最大時間步數（None 使用配置值）

        Returns:
            是否收斂
        """
        if self.flow is None:
            self.initialize_flow()

        if max_steps is None:
            max_steps = self.config.solver.max_steps

        for step in range(max_steps):
            self._step = step + 1

            # 執行時間步（殘差已在 _time_step 中記錄到 monitor）
            residual = self._time_step()

            # 計算質量流量
            mass_flow = self._compute_mass_flow()
            self.convergence_monitor.add_mass_flow(mass_flow)

            # 調用進度回調
            if self._progress_callback:
                self._progress_callback(self._step, residual, mass_flow)

            # 檢查收斂（使用 monitor）
            if self.convergence_monitor.is_converged():
                self._converged = True
                break

            # 檢查停滯
            if self.convergence_monitor.is_stalled():
                print(f"警告：求解在步 {step} 停滯")

            # 每 100 步輸出
            if step % 100 == 0:
                rate = self.convergence_monitor.get_convergence_rate()
                print(
                    f"步 {step}: 殘差 = {residual:.6e}, 流量 = {mass_flow:.4f}, 收斂率 = {rate:.4f}"
                )

        return self._converged

    def run(self, output_dir: str | Path | None = None) -> dict[str, object]:
        """執行完整計算並輸出結果。

        Args:
            output_dir: 輸出目錄（None 則不輸出）

        Returns:
            結果摘要
        """
        # 初始化
        self.initialize_flow()

        # 求解
        converged = self.solve()

        # 收集結果
        residual_history = self.convergence_monitor.residual_history
        mass_flow_history = self.convergence_monitor.mass_flow_history
        result: dict[str, object] = {
            "converged": converged,
            "steps": self._step,
            "final_residual": residual_history[-1] if residual_history else 1.0,
            "final_mass_flow": mass_flow_history[-1] if mass_flow_history else 0.0,
            "convergence_rate": self.convergence_monitor.get_convergence_rate(),
            "normalized_residual": self.convergence_monitor.normalized_residual,
        }

        # 輸出結果
        if output_dir:
            handler = MultallFileHandler(output_dir=output_dir)
            handler.write_outputs(self.config)
            result["output_dir"] = str(output_dir)

        return result

    def get_residual_history(self) -> list[float]:
        """獲取殘差歷史。

        Returns:
            殘差列表
        """
        return self.convergence_monitor.residual_history

    def get_mass_flow_history(self) -> list[float]:
        """獲取質量流量歷史。

        Returns:
            質量流量列表
        """
        return self.convergence_monitor.mass_flow_history

    @property
    def is_converged(self) -> bool:
        """是否已收斂。"""
        return self._converged

    @property
    def current_step(self) -> int:
        """當前時間步。"""
        return self._step


def create_simple_turbine_solver(
    inlet_po: float = 200000.0,
    inlet_to: float = 400.0,
    exit_ps: float = 100000.0,
    rpm: float = 10000.0,
    n_blades: int = 30,
) -> MultallSolver:
    """創建簡單渦輪求解器。

    Args:
        inlet_po: 進口總壓 [Pa]
        inlet_to: 進口總溫 [K]
        exit_ps: 出口靜壓 [Pa]
        rpm: 轉速 [RPM]
        n_blades: 葉片數

    Returns:
        配置好的求解器
    """
    from .data_structures import (
        BladeRowGeometry,
        ExitBoundary,
        GasProperties,
        GridParameters,
        InletBoundary,
        MixingPlaneParameters,
        SolverParameters,
        ViscousParameters,
    )

    config = MultallConfig(
        title="Simple Turbine",
        gas=GasProperties(cp=1005.0, gamma=1.4),
        grid=GridParameters(im=17, jm=50, km=9),
        solver=SolverParameters(
            max_steps=1000,
            cfl=0.5,
            convergence_limit=0.001,
        ),
        viscous=ViscousParameters(model=ViscousModel.MIXING_LENGTH),
        mixing_plane=MixingPlaneParameters(enabled=True),
        inlet=InletBoundary(
            use_total_pressure=True,
            po=[inlet_po] * 9,
            to=[inlet_to] * 9,
        ),
        exit=ExitBoundary(
            use_static_pressure=True,
            pstatic_hub=exit_ps,
            pstatic_tip=exit_ps,
        ),
        nrows=1,
        blade_rows=[
            BladeRowGeometry(
                row_number=1,
                row_type="R",
                n_blades=n_blades,
                rpm=rpm,
            )
        ],
    )

    return MultallSolver(config)
