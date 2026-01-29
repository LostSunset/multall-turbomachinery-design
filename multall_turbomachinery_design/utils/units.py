# -*- coding: utf-8 -*-
"""單位轉換工具。

提供渦輪機械設計常用的單位轉換功能。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class PressureUnit(Enum):
    """壓力單位。"""

    PA = "Pa"
    KPA = "kPa"
    MPA = "MPa"
    BAR = "bar"
    ATM = "atm"
    PSI = "psi"
    MMHG = "mmHg"


class TemperatureUnit(Enum):
    """溫度單位。"""

    K = "K"
    C = "°C"
    F = "°F"
    R = "°R"


class LengthUnit(Enum):
    """長度單位。"""

    M = "m"
    CM = "cm"
    MM = "mm"
    IN = "in"
    FT = "ft"


class VelocityUnit(Enum):
    """速度單位。"""

    M_S = "m/s"
    KM_H = "km/h"
    FT_S = "ft/s"
    MPH = "mph"
    KNOT = "knot"


class MassFlowUnit(Enum):
    """質量流率單位。"""

    KG_S = "kg/s"
    KG_H = "kg/h"
    LB_S = "lb/s"
    LB_H = "lb/h"


class AngularVelocityUnit(Enum):
    """角速度單位。"""

    RAD_S = "rad/s"
    RPM = "rpm"
    DEG_S = "deg/s"


class AngleUnit(Enum):
    """角度單位。"""

    RAD = "rad"
    DEG = "deg"


@dataclass
class UnitConverter:
    """單位轉換器。

    提供各種物理量的單位轉換。
    """

    # 壓力轉換因子（到 Pa）
    _pressure_to_pa: ClassVar[dict[PressureUnit, float]] = {
        PressureUnit.PA: 1.0,
        PressureUnit.KPA: 1000.0,
        PressureUnit.MPA: 1e6,
        PressureUnit.BAR: 1e5,
        PressureUnit.ATM: 101325.0,
        PressureUnit.PSI: 6894.757,
        PressureUnit.MMHG: 133.322,
    }

    # 長度轉換因子（到 m）
    _length_to_m: ClassVar[dict[LengthUnit, float]] = {
        LengthUnit.M: 1.0,
        LengthUnit.CM: 0.01,
        LengthUnit.MM: 0.001,
        LengthUnit.IN: 0.0254,
        LengthUnit.FT: 0.3048,
    }

    # 速度轉換因子（到 m/s）
    _velocity_to_ms: ClassVar[dict[VelocityUnit, float]] = {
        VelocityUnit.M_S: 1.0,
        VelocityUnit.KM_H: 1 / 3.6,
        VelocityUnit.FT_S: 0.3048,
        VelocityUnit.MPH: 0.44704,
        VelocityUnit.KNOT: 0.514444,
    }

    # 質量流率轉換因子（到 kg/s）
    _mass_flow_to_kgs: ClassVar[dict[MassFlowUnit, float]] = {
        MassFlowUnit.KG_S: 1.0,
        MassFlowUnit.KG_H: 1 / 3600,
        MassFlowUnit.LB_S: 0.453592,
        MassFlowUnit.LB_H: 0.453592 / 3600,
    }

    # 角速度轉換因子（到 rad/s）
    _angular_to_rads: ClassVar[dict[AngularVelocityUnit, float]] = {
        AngularVelocityUnit.RAD_S: 1.0,
        AngularVelocityUnit.RPM: math.pi / 30,
        AngularVelocityUnit.DEG_S: math.pi / 180,
    }

    @classmethod
    def convert_pressure(
        cls,
        value: float,
        from_unit: PressureUnit,
        to_unit: PressureUnit,
    ) -> float:
        """轉換壓力單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        pa = value * cls._pressure_to_pa[from_unit]
        return pa / cls._pressure_to_pa[to_unit]

    @classmethod
    def convert_temperature(
        cls,
        value: float,
        from_unit: TemperatureUnit,
        to_unit: TemperatureUnit,
    ) -> float:
        """轉換溫度單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        # 先轉換到 K
        if from_unit == TemperatureUnit.K:
            kelvin = value
        elif from_unit == TemperatureUnit.C:
            kelvin = value + 273.15
        elif from_unit == TemperatureUnit.F:
            kelvin = (value + 459.67) * 5 / 9
        else:  # R
            kelvin = value * 5 / 9

        # 再轉換到目標單位
        if to_unit == TemperatureUnit.K:
            return kelvin
        elif to_unit == TemperatureUnit.C:
            return kelvin - 273.15
        elif to_unit == TemperatureUnit.F:
            return kelvin * 9 / 5 - 459.67
        else:  # R
            return kelvin * 9 / 5

    @classmethod
    def convert_length(
        cls,
        value: float,
        from_unit: LengthUnit,
        to_unit: LengthUnit,
    ) -> float:
        """轉換長度單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        meters = value * cls._length_to_m[from_unit]
        return meters / cls._length_to_m[to_unit]

    @classmethod
    def convert_velocity(
        cls,
        value: float,
        from_unit: VelocityUnit,
        to_unit: VelocityUnit,
    ) -> float:
        """轉換速度單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        ms = value * cls._velocity_to_ms[from_unit]
        return ms / cls._velocity_to_ms[to_unit]

    @classmethod
    def convert_mass_flow(
        cls,
        value: float,
        from_unit: MassFlowUnit,
        to_unit: MassFlowUnit,
    ) -> float:
        """轉換質量流率單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        kgs = value * cls._mass_flow_to_kgs[from_unit]
        return kgs / cls._mass_flow_to_kgs[to_unit]

    @classmethod
    def convert_angular_velocity(
        cls,
        value: float,
        from_unit: AngularVelocityUnit,
        to_unit: AngularVelocityUnit,
    ) -> float:
        """轉換角速度單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        rads = value * cls._angular_to_rads[from_unit]
        return rads / cls._angular_to_rads[to_unit]

    @classmethod
    def convert_angle(
        cls,
        value: float,
        from_unit: AngleUnit,
        to_unit: AngleUnit,
    ) -> float:
        """轉換角度單位。

        Args:
            value: 數值
            from_unit: 來源單位
            to_unit: 目標單位

        Returns:
            轉換後的數值
        """
        if from_unit == to_unit:
            return value
        if from_unit == AngleUnit.RAD:
            return math.degrees(value)
        return math.radians(value)

    @classmethod
    def rpm_to_rad_s(cls, rpm: float) -> float:
        """RPM 轉 rad/s。"""
        return rpm * math.pi / 30

    @classmethod
    def rad_s_to_rpm(cls, rad_s: float) -> float:
        """rad/s 轉 RPM。"""
        return rad_s * 30 / math.pi

    @classmethod
    def bar_to_pa(cls, bar: float) -> float:
        """bar 轉 Pa。"""
        return bar * 1e5

    @classmethod
    def pa_to_bar(cls, pa: float) -> float:
        """Pa 轉 bar。"""
        return pa / 1e5

    @classmethod
    def celsius_to_kelvin(cls, celsius: float) -> float:
        """攝氏度轉開爾文。"""
        return celsius + 273.15

    @classmethod
    def kelvin_to_celsius(cls, kelvin: float) -> float:
        """開爾文轉攝氏度。"""
        return kelvin - 273.15


# 便捷函數
def convert_pressure(
    value: float,
    from_unit: str | PressureUnit,
    to_unit: str | PressureUnit,
) -> float:
    """轉換壓力單位的便捷函數。"""
    if isinstance(from_unit, str):
        from_unit = PressureUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = PressureUnit(to_unit)
    return UnitConverter.convert_pressure(value, from_unit, to_unit)


def convert_temperature(
    value: float,
    from_unit: str | TemperatureUnit,
    to_unit: str | TemperatureUnit,
) -> float:
    """轉換溫度單位的便捷函數。"""
    if isinstance(from_unit, str):
        from_unit = TemperatureUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = TemperatureUnit(to_unit)
    return UnitConverter.convert_temperature(value, from_unit, to_unit)


def convert_length(
    value: float,
    from_unit: str | LengthUnit,
    to_unit: str | LengthUnit,
) -> float:
    """轉換長度單位的便捷函數。"""
    if isinstance(from_unit, str):
        from_unit = LengthUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = LengthUnit(to_unit)
    return UnitConverter.convert_length(value, from_unit, to_unit)
