# -*- coding: utf-8 -*-
"""STAGEN 使用示例。

展示如何使用 STAGEN 進行 3D 葉片幾何生成。
"""

from __future__ import annotations

from multall_turbomachinery_design.stagen import (
    BladeProfileGenerator,
    StreamSurfaceProjector,
    ThicknessParameters,
)


def example_2d_blade_profile() -> None:
    """示例：生成 2D 葉片截面。"""
    print("=" * 60)
    print("示例 1：生成 2D 葉片截面")
    print("=" * 60)

    # 創建葉片截面生成器
    generator = BladeProfileGenerator()

    # 定義中弧線斜率（從進口到出口）
    x_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    camber_slope = [0.15, 0.08, 0.0, -0.08, -0.15]  # 斜率 dy/dx

    # 定義厚度參數
    thickness_params = ThicknessParameters(
        tk_le=0.02,  # 前緣厚度比 2%
        tk_te=0.01,  # 後緣厚度比 1%
        tk_max=0.10,  # 最大厚度比 10%
        xtk_max=0.40,  # 最大厚度位置在 40% 弦長處
        tk_type=2.0,  # 厚度分佈指數
        le_exp=3.0,  # 前緣橢圓化指數
        xmod_le=0.02,  # 前緣修正範圍
        xmod_te=0.01,  # 後緣修正範圍
        f_perp=1.0,  # 厚度垂直於中弧線
    )

    # 生成葉片截面
    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=thickness_params,
        npoints=200,
    )

    # 打印結果
    print("\n生成的葉片截面：")
    print(f"  點數：{len(profile.x_camber)}")
    print(f"  弦長：{profile.chord_length}")
    print(f"  前緣位置：X = {profile.leading_edge_x:.3f}")
    print(f"  後緣位置：X = {profile.trailing_edge_x:.3f}")

    # 打印中弧線範圍
    y_min = min(profile.y_camber)
    y_max = max(profile.y_camber)
    print("\n中弧線範圍：")
    print(f"  Y 最小值：{y_min:.4f}")
    print(f"  Y 最大值：{y_max:.4f}")

    # 打印厚度範圍
    tk_min = min(profile.thickness)
    tk_max = max(profile.thickness)
    print("\n厚度範圍：")
    print(f"  最小厚度：{tk_min:.4f}")
    print(f"  最大厚度：{tk_max:.4f}")

    print()


def example_stream_surface_creation() -> None:
    """示例：創建流線表面。"""
    print("=" * 60)
    print("示例 2：創建流線表面")
    print("=" * 60)

    # 創建流線投影器
    projector = StreamSurfaceProjector()

    # 定義軸向流流線表面（半徑恆定）
    x_coords = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    r_coords = [0.30, 0.30, 0.30, 0.30, 0.30, 0.30]

    # 創建流線表面
    surface = projector.create_stream_surface(x_coords, r_coords)

    print("\n軸向流流線表面：")
    print(f"  點數：{surface.npoints}")
    print(f"  軸向範圍：X = {min(surface.x):.3f} ~ {max(surface.x):.3f} m")
    print(f"  半徑：R = {surface.r[0]:.3f} m（恆定）")
    print(f"  子午線長度：{surface.s_meridional[-1]:.4f} m")

    # 定位前後緣
    le_x = 0.02
    te_x = 0.08
    projector.locate_leading_trailing_edges(surface, le_x, te_x)

    print("\n前後緣位置：")
    print(f"  前緣：X = {surface.le_x:.3f} m, R = {surface.le_r:.3f} m")
    print(f"  後緣：X = {surface.te_x:.3f} m, R = {surface.te_r:.3f} m")
    print(f"  子午弦長：{surface.chord_meridional:.4f} m")

    # 創建混流流線表面（半徑變化）
    x_coords_mixed = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    r_coords_mixed = [0.28, 0.29, 0.30, 0.31, 0.32, 0.33]

    surface_mixed = projector.create_stream_surface(x_coords_mixed, r_coords_mixed)

    print("\n混流流線表面：")
    print(f"  點數：{surface_mixed.npoints}")
    print(f"  軸向範圍：X = {min(surface_mixed.x):.3f} ~ {max(surface_mixed.x):.3f} m")
    print(f"  半徑範圍：R = {min(surface_mixed.r):.3f} ~ {max(surface_mixed.r):.3f} m")
    print(f"  子午線長度：{surface_mixed.s_meridional[-1]:.4f} m")

    print()


def example_3d_projection() -> None:
    """示例：將 2D 葉片投影到 3D 流線表面。"""
    print("=" * 60)
    print("示例 3：2D 葉片投影到 3D 流線表面")
    print("=" * 60)

    # 創建生成器和投影器
    generator = BladeProfileGenerator()
    projector = StreamSurfaceProjector()

    # 生成 2D 葉片截面
    x_fractions = [0.0, 0.5, 1.0]
    camber_slope = [0.1, 0.0, -0.1]
    thickness_params = ThicknessParameters(
        tk_le=0.02, tk_te=0.01, tk_max=0.08, xtk_max=0.40
    )

    profile = generator.generate_from_camber_thickness(
        camber_slope=camber_slope,
        x_fractions=x_fractions,
        thickness_params=thickness_params,
        npoints=100,
    )

    # 創建流線表面
    x_coords = [0.0, 0.025, 0.05, 0.075, 0.10]
    r_coords = [0.30, 0.30, 0.30, 0.30, 0.30]
    surface = projector.create_stream_surface(x_coords, r_coords)
    projector.locate_leading_trailing_edges(surface, 0.02, 0.08)

    # 投影到流線表面
    section = projector.project_profile_to_surface(
        profile=profile,
        surface=surface,
        section_number=1,
        spanwise_fraction=0.0,  # HUB 截面
    )

    print("\n投影結果：")
    print(f"  截面號：{section.section_number}")
    print(f"  跨向位置：{section.spanwise_fraction * 100:.0f}%")
    print(f"  網格點數：{len(section.x_grid)}")

    # 計算質心
    projector.calculate_centroid(section)
    print("\n質心位置：")
    print(f"  X = {section.x_centroid:.4f} m")
    print(f"  Y = {section.y_centroid:.4f} m")

    # R-THETA 轉換
    projector.convert_r_theta_to_cartesian(section)
    print("\n坐標轉換：")
    print("  已轉換為笛卡爾坐標系（相對於質心）")

    print()


def example_stacking_transformation() -> None:
    """示例：3D 堆疊變換。"""
    print("=" * 60)
    print("示例 4：3D 堆疊變換")
    print("=" * 60)

    from multall_turbomachinery_design.stagen import (
        BladeSection3D,
        StackingParameters,
    )

    # 創建投影器
    projector = StreamSurfaceProjector()

    # 創建簡單截面
    section = BladeSection3D(
        section_number=2,
        spanwise_fraction=0.5,  # 中跨截面
        x_grid=[0.0, 0.025, 0.05],
        y_grid=[0.0, 0.0, 0.0],
        r_grid=[0.35, 0.35, 0.35],
        tk_grid=[0.02, 0.02, 0.02],
    )
    section.j_le = 0
    section.j_te = 2
    section.x_centroid = 0.025
    section.y_centroid = 0.0

    print("\n原始截面：")
    print(f"  前緣位置：X = {section.x_grid[0]:.3f} m")
    print(f"  後緣位置：X = {section.x_grid[-1]:.3f} m")
    print(f"  質心位置：X = {section.x_centroid:.3f} m")

    # 定義堆疊參數
    stacking = StackingParameters(
        f_centroid=0.5,  # 50% 向 HUB 中心堆疊
        f_tang=0.1,  # 10% 弦長的周向移動
        f_lean=0.05,  # 5% 弦長的傾斜
        f_sweep=0.02,  # 2% 弦長的掃蕩
        f_scale=1.1,  # 110% 縮放
        f_const=0.0,  # 固定點在前緣
        x_centroid_hub=0.020,  # HUB 質心位置
        y_centroid_hub=0.001,
    )

    # 創建流線表面
    x_coords = [0.0, 0.025, 0.05, 0.075, 0.10]
    r_coords = [0.35, 0.35, 0.35, 0.35, 0.35]
    surface = projector.create_stream_surface(x_coords, r_coords)

    # 應用堆疊變換
    projector.apply_stacking(section, stacking, surface)

    print("\n堆疊變換後：")
    print(f"  前緣位置：X = {section.x_grid[0]:.3f} m")
    print(f"  後緣位置：X = {section.x_grid[-1]:.3f} m")
    print(f"  質心位置：X = {section.x_centroid:.3f} m, Y = {section.y_centroid:.3f} m")

    print("\n堆疊效果：")
    print(f"  周向移動（f_tang）：{stacking.f_tang * 100:.0f}% 弦長")
    print(f"  傾斜（f_lean）：{stacking.f_lean * 100:.0f}% 弦長")
    print(f"  掃蕩（f_sweep）：{stacking.f_sweep * 100:.0f}% 弦長")
    print(f"  縮放比例：{stacking.f_scale * 100:.0f}%")

    print()


if __name__ == "__main__":
    print("\nMULTALL STAGEN - 3D 葉片幾何生成工具")
    print("=" * 60)
    print()

    # 運行所有示例
    example_2d_blade_profile()
    example_stream_surface_creation()
    example_3d_projection()
    example_stacking_transformation()

    print("所有示例完成！")
    print()
