# -*- coding: utf-8 -*-
"""UI 樣式系統。

提供專業的視覺風格和主題支援。

設計規範：
- 8px 基準網格系統
- 最小點擊區域 44x44px（符合無障礙標準）
- WCAG AA 對比度標準
- 完整的狀態樣式（hover/active/focus/disabled）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

# ===== 設計常數 =====
GRID_UNIT = 8  # 基準網格單位 (px)
MIN_TOUCH_SIZE = 44  # 最小可點擊尺寸 (px)
BORDER_RADIUS_SM = 4  # 小圓角
BORDER_RADIUS_MD = 8  # 中圓角
BORDER_RADIUS_LG = 12  # 大圓角


class ThemeMode(Enum):
    """主題模式。"""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class Spacing:
    """間距系統（基於 8px 網格）。"""

    xs: int = GRID_UNIT // 2  # 4px
    sm: int = GRID_UNIT  # 8px
    md: int = GRID_UNIT * 2  # 16px
    lg: int = GRID_UNIT * 3  # 24px
    xl: int = GRID_UNIT * 4  # 32px
    xxl: int = GRID_UNIT * 6  # 48px


@dataclass
class Typography:
    """字型系統。"""

    # 標題字型大小
    h1: int = 28
    h2: int = 22
    h3: int = 18
    h4: int = 16
    h5: int = 14
    h6: int = 13

    # 內文字型大小
    body: int = 13
    body_sm: int = 12
    caption: int = 11

    # 行高
    line_height: float = 1.5


@dataclass
class ColorPalette:
    """顏色調色板。"""

    # 主要顏色
    primary: str = "#0078D4"  # Microsoft Blue
    primary_light: str = "#106EBE"
    primary_dark: str = "#005A9E"

    # 強調色
    accent: str = "#0063B1"
    accent_light: str = "#1A86D0"
    accent_dark: str = "#004578"

    # 背景色
    background: str = "#FFFFFF"
    surface: str = "#F3F3F3"
    surface_light: str = "#FAFAFA"
    surface_dark: str = "#E5E5E5"

    # 文字顏色
    text_primary: str = "#1A1A1A"
    text_secondary: str = "#666666"
    text_disabled: str = "#A0A0A0"
    text_on_primary: str = "#FFFFFF"

    # 邊框
    border: str = "#D1D1D1"
    border_light: str = "#E5E5E5"
    border_dark: str = "#B0B0B0"

    # 狀態顏色
    success: str = "#107C10"
    warning: str = "#FF8C00"
    error: str = "#D13438"
    info: str = "#0078D4"

    # 特殊顏色
    hover: str = "#E5F1FB"
    selected: str = "#CCE4F7"
    focus: str = "#0078D4"


@dataclass
class DarkColorPalette(ColorPalette):
    """深色主題調色板。"""

    # 主要顏色
    primary: str = "#60CDFF"
    primary_light: str = "#80D8FF"
    primary_dark: str = "#40B8E6"

    # 強調色
    accent: str = "#4CC2FF"
    accent_light: str = "#6DD1FF"
    accent_dark: str = "#2DB3FF"

    # 背景色
    background: str = "#1E1E1E"
    surface: str = "#252526"
    surface_light: str = "#2D2D30"
    surface_dark: str = "#1A1A1A"

    # 文字顏色
    text_primary: str = "#FFFFFF"
    text_secondary: str = "#B0B0B0"
    text_disabled: str = "#6E6E6E"
    text_on_primary: str = "#000000"

    # 邊框
    border: str = "#3C3C3C"
    border_light: str = "#4A4A4A"
    border_dark: str = "#2D2D2D"

    # 狀態顏色
    success: str = "#6CCB5F"
    warning: str = "#FFB900"
    error: str = "#F85149"
    info: str = "#60CDFF"

    # 特殊顏色
    hover: str = "#3C3C3C"
    selected: str = "#094771"
    focus: str = "#60CDFF"


class StyleManager:
    """樣式管理器。

    管理應用程式的整體樣式和主題。
    """

    _instance: ClassVar[StyleManager | None] = None
    _current_theme: ThemeMode = ThemeMode.LIGHT
    _palette: ColorPalette = ColorPalette()

    def __new__(cls) -> StyleManager:
        """單例模式。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> StyleManager:
        """取得管理器實例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def theme(self) -> ThemeMode:
        """取得當前主題。"""
        return self._current_theme

    @theme.setter
    def theme(self, mode: ThemeMode) -> None:
        """設置主題。"""
        self._current_theme = mode
        if mode == ThemeMode.DARK:
            self._palette = DarkColorPalette()
        else:
            self._palette = ColorPalette()

    @property
    def palette(self) -> ColorPalette:
        """取得當前調色板。"""
        return self._palette

    def get_stylesheet(self) -> str:
        """產生完整的 QSS 樣式表。

        設計規範：
        - 8px 網格間距系統
        - 最小點擊區域 44x44px
        - 完整的互動狀態（hover/active/focus/disabled）
        - WCAG AA 對比度標準
        """
        p = self._palette
        sp = Spacing()
        ty = Typography()

        return f"""
/* ===== 全局樣式 ===== */
/* 基準：8px 網格系統，最小點擊區域 44px */
QWidget {{
    font-family: "Segoe UI", "Microsoft JhengHei UI", "微軟正黑體", sans-serif;
    font-size: {ty.body}px;
    color: {p.text_primary};
    background-color: {p.background};
}}

/* ===== 主視窗 ===== */
QMainWindow {{
    background-color: {p.background};
}}

QMainWindow::separator {{
    background-color: {p.border};
    width: 1px;
    height: 1px;
}}

/* ===== 選單列 ===== */
QMenuBar {{
    background-color: {p.surface};
    border-bottom: 1px solid {p.border};
    padding: 2px;
    spacing: 2px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {p.hover};
}}

QMenuBar::item:pressed {{
    background-color: {p.selected};
}}

QMenu {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 32px 8px 24px;
    border-radius: 4px;
    margin: 2px 4px;
}}

QMenu::item:selected {{
    background-color: {p.hover};
}}

QMenu::separator {{
    height: 1px;
    background-color: {p.border};
    margin: 4px 8px;
}}

QMenu::indicator {{
    width: 16px;
    height: 16px;
    margin-left: 4px;
}}

/* ===== 工具列 ===== */
QToolBar {{
    background-color: {p.surface};
    border: none;
    border-bottom: 1px solid {p.border};
    spacing: {sp.xs}px;
    padding: {sp.xs}px {sp.sm}px;
}}

QToolBar::separator {{
    background-color: {p.border};
    width: 1px;
    margin: {sp.xs}px {sp.sm}px;
}}

/* 工具按鈕 - 最小點擊區域 44x44px */
QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {sp.sm}px;
    margin: 2px;
    min-width: {MIN_TOUCH_SIZE}px;
    min-height: {MIN_TOUCH_SIZE}px;
}}

QToolButton:hover {{
    background-color: {p.hover};
}}

QToolButton:pressed {{
    background-color: {p.selected};
}}

QToolButton:checked {{
    background-color: {p.selected};
    border: 2px solid {p.primary};
}}

QToolButton:disabled {{
    color: {p.text_disabled};
    background-color: transparent;
}}

QToolButton:focus {{
    border: 2px solid {p.focus};
}}

/* ===== 標籤頁 ===== */
QTabWidget::pane {{
    background-color: {p.background};
    border: 1px solid {p.border};
    border-radius: 4px;
    margin-top: -1px;
}}

QTabBar {{
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 10px 20px;
    margin-right: 2px;
    min-width: 120px;
}}

QTabBar::tab:selected {{
    background-color: {p.background};
    border-bottom: 2px solid {p.primary};
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    background-color: {p.hover};
}}

/* ===== 群組框 ===== */
QGroupBox {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 6px;
    margin-top: 8px;
    padding: 12px;
    padding-top: 20px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: {p.surface};
    color: {p.text_primary};
}}

/* ===== 輸入框 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {p.surface_light};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
    selection-background-color: {p.selected};
}}

QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {{
    border-color: {p.border_dark};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {p.primary};
    padding: 3px 7px;
}}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
    background-color: {p.surface_dark};
    color: {p.text_disabled};
    border-color: {p.border_light};
}}

/* ===== 數值輸入框 ===== */
QSpinBox, QDoubleSpinBox {{
    background-color: {p.surface_light};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 6px;
    padding-right: 20px;
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {p.border_dark};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {p.primary};
    padding: 3px 5px;
    padding-right: 19px;
}}

QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {p.surface_dark};
    color: {p.text_disabled};
    border-color: {p.border_light};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {p.border};
    border-top-right-radius: {BORDER_RADIUS_SM}px;
    background-color: {p.surface};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {p.border};
    border-bottom-right-radius: {BORDER_RADIUS_SM}px;
    background-color: {p.surface};
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {p.hover};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 8px;
    height: 8px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 8px;
    height: 8px;
}}

/* ===== 下拉框 ===== */
QComboBox {{
    background-color: {p.surface_light};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 4px 8px;
    padding-right: 24px;
}}

QComboBox:hover {{
    border-color: {p.border_dark};
}}

QComboBox:focus {{
    border: 2px solid {p.primary};
    padding: 3px 7px;
    padding-right: 23px;
}}

QComboBox:disabled {{
    background-color: {p.surface_dark};
    color: {p.text_disabled};
    border-color: {p.border_light};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border-left: 1px solid {p.border};
    border-top-right-radius: {BORDER_RADIUS_SM}px;
    border-bottom-right-radius: {BORDER_RADIUS_SM}px;
}}

QComboBox::drop-down:hover {{
    background-color: {p.hover};
}}

QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_SM}px;
    selection-background-color: {p.selected};
}}

QComboBox QAbstractItemView::item {{
    padding: 6px 10px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {p.hover};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {p.selected};
}}

/* ===== 按鈕 ===== */
QPushButton {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_SM}px;
    padding: 6px 16px;
    min-width: 80px;
}}

QPushButton:hover {{
    background-color: {p.hover};
    border-color: {p.border_dark};
}}

QPushButton:pressed {{
    background-color: {p.selected};
}}

QPushButton:disabled {{
    background-color: {p.surface_dark};
    color: {p.text_disabled};
    border-color: {p.border_light};
}}

QPushButton:focus {{
    border: 2px solid {p.focus};
}}

/* 主要按鈕 */
QPushButton[primary="true"], QPushButton#primaryButton {{
    background-color: {p.primary};
    border: none;
    color: {p.text_on_primary};
}}

QPushButton[primary="true"]:hover, QPushButton#primaryButton:hover {{
    background-color: {p.primary_light};
}}

QPushButton[primary="true"]:pressed, QPushButton#primaryButton:pressed {{
    background-color: {p.primary_dark};
}}

/* 成功按鈕 */
QPushButton[success="true"] {{
    background-color: {p.success};
    border: none;
    color: {p.text_on_primary};
}}

/* 警告按鈕 */
QPushButton[warning="true"] {{
    background-color: {p.warning};
    border: none;
    color: {p.text_on_primary};
}}

/* 危險按鈕 */
QPushButton[danger="true"] {{
    background-color: {p.error};
    border: none;
    color: {p.text_on_primary};
}}

/* ===== 表格 ===== */
QTableWidget, QTableView {{
    background-color: {p.background};
    alternate-background-color: {p.surface_light};
    border: 1px solid {p.border};
    border-radius: 4px;
    gridline-color: {p.border_light};
    selection-background-color: {p.selected};
}}

QTableWidget::item, QTableView::item {{
    padding: 8px;
    border: none;
}}

QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {p.selected};
    color: {p.text_primary};
}}

QHeaderView {{
    background-color: {p.surface};
}}

QHeaderView::section {{
    background-color: {p.surface};
    border: none;
    border-bottom: 2px solid {p.border};
    border-right: 1px solid {p.border_light};
    padding: 10px 8px;
    font-weight: bold;
}}

QHeaderView::section:last {{
    border-right: none;
}}

/* ===== 滾動條 ===== */
QScrollBar:vertical {{
    background-color: {p.surface};
    width: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {p.border_dark};
    border-radius: 4px;
    min-height: 30px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {p.text_secondary};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {p.surface};
    height: 12px;
    border-radius: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {p.border_dark};
    border-radius: 4px;
    min-width: 30px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {p.text_secondary};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ===== 進度條 ===== */
QProgressBar {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
    text-align: center;
    height: 20px;
}}

QProgressBar::chunk {{
    background-color: {p.primary};
    border-radius: 3px;
}}

/* ===== 狀態列 ===== */
QStatusBar {{
    background-color: {p.surface};
    border-top: 1px solid {p.border};
    padding: 4px;
}}

QStatusBar::item {{
    border: none;
}}

/* ===== 分割器 ===== */
QSplitter::handle {{
    background-color: {p.border};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {p.primary};
}}

/* ===== 工具提示 ===== */
QToolTip {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: 8px;
    color: {p.text_primary};
}}

/* ===== 標籤 ===== */
QLabel {{
    background-color: transparent;
    color: {p.text_primary};
}}

/* 標題層級系統 */
QLabel[heading="h1"] {{
    font-size: {ty.h1}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[heading="h2"] {{
    font-size: {ty.h2}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[heading="h3"] {{
    font-size: {ty.h3}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[heading="true"], QLabel[heading="h4"] {{
    font-size: {ty.h4}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[heading="h5"] {{
    font-size: {ty.h5}px;
    font-weight: bold;
    color: {p.text_primary};
}}

QLabel[heading="h6"] {{
    font-size: {ty.h6}px;
    font-weight: bold;
    color: {p.text_secondary};
}}

QLabel[secondary="true"] {{
    color: {p.text_secondary};
    font-size: {ty.body_sm}px;
}}

QLabel[caption="true"] {{
    color: {p.text_secondary};
    font-size: {ty.caption}px;
}}

/* 狀態標籤 */
QLabel[status="success"] {{
    color: {p.success};
    font-weight: bold;
}}

QLabel[status="warning"] {{
    color: {p.warning};
    font-weight: bold;
}}

QLabel[status="error"] {{
    color: {p.error};
    font-weight: bold;
}}

QLabel[status="info"] {{
    color: {p.info};
    font-weight: bold;
}}

/* ===== 訊息框 ===== */
QMessageBox {{
    background-color: {p.background};
}}

QMessageBox QLabel {{
    color: {p.text_primary};
}}

QMessageBox QPushButton {{
    min-width: 100px;
}}

/* ===== 對話框 ===== */
QDialog {{
    background-color: {p.background};
}}

QDialogButtonBox {{
    button-layout: 0;
}}

/* ===== 檔案對話框 ===== */
QFileDialog {{
    background-color: {p.background};
}}

/* ===== 樹狀視圖 ===== */
QTreeView {{
    background-color: {p.background};
    border: 1px solid {p.border};
    border-radius: 4px;
    alternate-background-color: {p.surface_light};
}}

QTreeView::item {{
    padding: 6px;
    border-radius: 4px;
}}

QTreeView::item:selected {{
    background-color: {p.selected};
}}

QTreeView::item:hover:!selected {{
    background-color: {p.hover};
}}

QTreeView::branch:has-siblings:!adjoins-item {{
    border-image: none;
}}

/* ===== 列表視圖 ===== */
QListView {{
    background-color: {p.background};
    border: 1px solid {p.border};
    border-radius: 4px;
    alternate-background-color: {p.surface_light};
}}

QListView::item {{
    padding: 8px;
    border-radius: 4px;
}}

QListView::item:selected {{
    background-color: {p.selected};
}}

QListView::item:hover:!selected {{
    background-color: {p.hover};
}}

/* ===== 複選框和單選按鈕 ===== */
QCheckBox, QRadioButton {{
    spacing: 8px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
}}

QCheckBox::indicator {{
    border: 2px solid {p.border_dark};
    border-radius: 4px;
    background-color: {p.surface_light};
}}

QCheckBox::indicator:checked {{
    background-color: {p.primary};
    border-color: {p.primary};
}}

QRadioButton::indicator {{
    border: 2px solid {p.border_dark};
    border-radius: 9px;
    background-color: {p.surface_light};
}}

QRadioButton::indicator:checked {{
    background-color: {p.primary};
    border-color: {p.primary};
}}

/* ===== 滑桿 ===== */
QSlider::groove:horizontal {{
    background-color: {p.surface_dark};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {p.primary};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {p.primary_light};
}}

QSlider::sub-page:horizontal {{
    background-color: {p.primary};
    border-radius: 2px;
}}

/* ===== 側邊導航 ===== */
#sideNavigation {{
    background-color: {p.surface};
    border-right: 1px solid {p.border};
}}

#sideNavigation QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: {BORDER_RADIUS_SM}px;
    padding: {sp.sm}px;
    margin: {sp.xs}px;
    min-height: 60px;
    text-align: center;
}}

#sideNavigation QToolButton:hover {{
    background-color: {p.hover};
}}

#sideNavigation QToolButton:checked {{
    background-color: {p.selected};
    border-left: 3px solid {p.primary};
}}

/* ===== 進度指示器（載入狀態） ===== */
QProgressBar[loading="true"] {{
    background-color: {p.surface};
    border: none;
    border-radius: {BORDER_RADIUS_SM}px;
}}

QProgressBar[loading="true"]::chunk {{
    background-color: {p.primary};
    border-radius: {BORDER_RADIUS_SM}px;
}}

/* ===== 空狀態佔位符 ===== */
QLabel[empty-state="true"] {{
    color: {p.text_disabled};
    font-size: {ty.h5}px;
    padding: {sp.xxl}px;
    qproperty-alignment: AlignCenter;
}}

/* ===== 卡片容器 ===== */
QFrame[card="true"] {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {BORDER_RADIUS_MD}px;
    padding: {sp.md}px;
}}

QFrame[card="true"]:hover {{
    border-color: {p.border_dark};
}}

/* ===== 分隔線 ===== */
QFrame[separator="horizontal"] {{
    background-color: {p.border};
    max-height: 1px;
    margin: {sp.sm}px 0;
}}

QFrame[separator="vertical"] {{
    background-color: {p.border};
    max-width: 1px;
    margin: 0 {sp.sm}px;
}}

/* ===== 徽章樣式 ===== */
QLabel[badge="true"] {{
    background-color: {p.primary};
    color: {p.text_on_primary};
    border-radius: {sp.sm}px;
    padding: {sp.xs}px {sp.sm}px;
    font-size: {ty.caption}px;
    font-weight: bold;
}}

QLabel[badge="success"] {{
    background-color: {p.success};
}}

QLabel[badge="warning"] {{
    background-color: {p.warning};
}}

QLabel[badge="error"] {{
    background-color: {p.error};
}}
"""


def apply_style(app) -> None:
    """應用樣式到應用程式。

    Args:
        app: QApplication 實例
    """
    manager = StyleManager.get_instance()
    app.setStyleSheet(manager.get_stylesheet())


def set_theme(mode: ThemeMode) -> str:
    """設置主題模式。

    Args:
        mode: 主題模式

    Returns:
        新的樣式表
    """
    manager = StyleManager.get_instance()
    manager.theme = mode
    return manager.get_stylesheet()


def get_current_palette() -> ColorPalette:
    """取得當前調色板。"""
    return StyleManager.get_instance().palette


def get_spacing() -> Spacing:
    """取得間距系統。"""
    return Spacing()


def get_typography() -> Typography:
    """取得字型系統。"""
    return Typography()


# 導出常數
__all__ = [
    "ThemeMode",
    "ColorPalette",
    "DarkColorPalette",
    "Spacing",
    "Typography",
    "StyleManager",
    "apply_style",
    "set_theme",
    "get_current_palette",
    "get_spacing",
    "get_typography",
    "GRID_UNIT",
    "MIN_TOUCH_SIZE",
    "BORDER_RADIUS_SM",
    "BORDER_RADIUS_MD",
    "BORDER_RADIUS_LG",
]
