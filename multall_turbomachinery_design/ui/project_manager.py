# -*- coding: utf-8 -*-
"""專案管理器。

提供專案的載入和儲存功能。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from multall_turbomachinery_design import __version__


@dataclass
class ProjectMetadata:
    """專案元資料。"""

    name: str = "未命名專案"
    description: str = ""
    author: str = ""
    created_at: str = ""
    modified_at: str = ""
    version: str = __version__


@dataclass
class ProjectData:
    """專案資料。"""

    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    meangen: dict[str, Any] = field(default_factory=dict)
    stagen: dict[str, Any] = field(default_factory=dict)
    multall: dict[str, Any] = field(default_factory=dict)


class ProjectManager:
    """專案管理器。

    負責專案的載入、儲存和管理。

    Example:
        >>> manager = ProjectManager()
        >>> manager.new_project("我的渦輪設計")
        >>> manager.data.meangen = panel.get_state()
        >>> manager.save("project.mtproj")
    """

    def __init__(self) -> None:
        """初始化專案管理器。"""
        self._data: ProjectData | None = None
        self._current_file: Path | None = None
        self._is_modified: bool = False

    @property
    def data(self) -> ProjectData | None:
        """獲取當前專案資料。"""
        return self._data

    @property
    def current_file(self) -> Path | None:
        """獲取當前專案檔案路徑。"""
        return self._current_file

    @property
    def is_modified(self) -> bool:
        """檢查專案是否有未儲存的變更。"""
        return self._is_modified

    @property
    def has_project(self) -> bool:
        """檢查是否有開啟的專案。"""
        return self._data is not None

    def new_project(self, name: str = "未命名專案") -> ProjectData:
        """建立新專案。

        Args:
            name: 專案名稱

        Returns:
            新建的專案資料
        """
        now = datetime.now().isoformat()
        self._data = ProjectData(
            metadata=ProjectMetadata(
                name=name,
                created_at=now,
                modified_at=now,
                version=__version__,
            )
        )
        self._current_file = None
        self._is_modified = False
        return self._data

    def load(self, file_path: str | Path) -> ProjectData:
        """載入專案。

        Args:
            file_path: 專案檔案路徑

        Returns:
            載入的專案資料

        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 檔案格式錯誤
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"專案檔案不存在: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無效的專案檔案格式: {e}") from e

        # 驗證檔案格式
        if "metadata" not in raw_data:
            raise ValueError("無效的專案檔案：缺少 metadata")

        # 解析元資料
        metadata = ProjectMetadata(**raw_data.get("metadata", {}))

        # 建立專案資料
        self._data = ProjectData(
            metadata=metadata,
            meangen=raw_data.get("meangen", {}),
            stagen=raw_data.get("stagen", {}),
            multall=raw_data.get("multall", {}),
        )

        self._current_file = file_path
        self._is_modified = False

        return self._data

    def save(self, file_path: str | Path | None = None) -> Path:
        """儲存專案。

        Args:
            file_path: 專案檔案路徑（None 使用當前檔案）

        Returns:
            儲存的檔案路徑

        Raises:
            ValueError: 沒有開啟的專案或未指定檔案路徑
        """
        if self._data is None:
            raise ValueError("沒有開啟的專案")

        if file_path is None:
            if self._current_file is None:
                raise ValueError("未指定儲存路徑")
            file_path = self._current_file
        else:
            file_path = Path(file_path)

        # 更新修改時間
        self._data.metadata.modified_at = datetime.now().isoformat()

        # 確保副檔名
        if not file_path.suffix:
            file_path = file_path.with_suffix(".mtproj")

        # 確保目錄存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 序列化資料
        data_dict = {
            "metadata": asdict(self._data.metadata),
            "meangen": self._data.meangen,
            "stagen": self._data.stagen,
            "multall": self._data.multall,
        }

        # 寫入檔案
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

        self._current_file = file_path
        self._is_modified = False

        return file_path

    def mark_modified(self) -> None:
        """標記專案已修改。"""
        self._is_modified = True
        if self._data is not None:
            self._data.metadata.modified_at = datetime.now().isoformat()

    def update_meangen(self, data: dict[str, Any]) -> None:
        """更新 MEANGEN 資料。

        Args:
            data: MEANGEN 參數資料
        """
        if self._data is not None:
            self._data.meangen = data
            self.mark_modified()

    def update_stagen(self, data: dict[str, Any]) -> None:
        """更新 STAGEN 資料。

        Args:
            data: STAGEN 參數資料
        """
        if self._data is not None:
            self._data.stagen = data
            self.mark_modified()

    def update_multall(self, data: dict[str, Any]) -> None:
        """更新 MULTALL 資料。

        Args:
            data: MULTALL 參數資料
        """
        if self._data is not None:
            self._data.multall = data
            self.mark_modified()

    def get_project_info(self) -> dict[str, str]:
        """獲取專案資訊。

        Returns:
            專案資訊字典
        """
        if self._data is None:
            return {}

        return {
            "name": self._data.metadata.name,
            "description": self._data.metadata.description,
            "author": self._data.metadata.author,
            "created": self._data.metadata.created_at,
            "modified": self._data.metadata.modified_at,
            "version": self._data.metadata.version,
            "file": str(self._current_file) if self._current_file else "(未儲存)",
        }
