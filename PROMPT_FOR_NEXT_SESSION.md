# 給下一個 Claude 的提示詞

## 🎯 專案接續說明

你正在繼續開發 **MULTALL 渦輪機械設計系統**，這是一個基於 Python 3.14 和 PySide6 的現代化渦輪機械設計系統。

**GitHub 倉庫**: https://github.com/LostSunset/multall-turbomachinery-design

## 📋 核心要求

### 1. 開發原則
- **自動持續開發**: 不需要詢問用戶，直接繼續開發
- **GitHub 推送**: 每次完成開發後**必須**推送到 GitHub
- **語言**: 所有溝通、註釋、文檔使用**正體中文**

### 2. 最終目標
輸出完整 3D 葉片 CAD，包含：
- 流道（shroud、hub）
- Rotors（轉子）
- Stators（定子）

### 3. 當前狀態
- MEANGEN 模組：✅ 100% 完成
- STAGEN 模組：🚧 85% 完成
  - ✅ 數據結構、2D 生成、3D 投影、網格生成、I/O 處理
  - ⏳ **主求解器待完成** ← 你的首要任務
- MULTALL 模組：🔮 規劃中
- 測試：113 個全部通過，94% 覆蓋率

## 🚀 你的第一步

```bash
# 1. 確認環境
cd /mnt/d/11_Multall_Turbomachinery_Design  # 或在 Windows 上對應的路徑
git status
git log --oneline -5  # 查看最近的提交

# 2. 閱讀交接手冊
cat HANDOVER.md  # 或使用 Read 工具

# 3. 開始開發
# 建議從 STAGEN 主求解器開始（stagen/solver.py）
```

## 📝 Git 推送規則

### Commit Message 格式

```
類型(範圍): 簡短描述（50字內）

詳細描述（可選）：
- 做了什麼
- 為什麼這樣做
- 測試結果

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### 類型說明
- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 代碼風格
- `refactor`: 重構
- `test`: 測試相關

### 範圍說明
- `meangen`: MEANGEN 模組
- `stagen`: STAGEN 模組
- `multall`: MULTALL 模組
- `deps`: 依賴
- `ci`: CI/CD

### 推送工作流

```bash
# 1. 檢查並修復代碼風格
.venv/bin/python -m ruff check . --fix --unsafe-fixes
.venv/bin/python -m ruff format .

# 2. 運行測試
.venv/bin/python -m pytest tests/ --cov

# 3. 添加文件
git add <files>

# 4. 提交
git commit -m "feat(stagen): 實現主求解器

新增功能：
- 整合所有 STAGEN 組件
- 完整的 3D 葉片生成流程

測試：
- 新增 15 個測試，全部通過
- 覆蓋率 97%

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 5. 推送
git push origin main
```

## ⚠️ 重要注意事項

### 1. 虛擬環境
```bash
# Linux/WSL
source .venv/bin/activate
.venv/bin/python -m pytest

# Windows
.venv\Scripts\activate
.venv\Scripts\python -m pytest
```

### 2. 依賴安裝
```bash
# ✅ 正確
uv sync --extra dev

# ❌ 錯誤（會失敗）
uv sync --all-extras  # CAD 依賴不支援 Python 3.14
```

### 3. 測試要求
- 所有新功能必須有測試
- 測試必須全部通過
- 覆蓋率不能降低

### 4. 代碼風格
- 使用 Ruff 進行檢查和格式化
- 所有檢查必須通過才能推送

## 🎯 優先開發任務

### Priority 1: STAGEN 主求解器
創建 `multall_turbomachinery_design/stagen/solver.py`:

```python
# -*- coding: utf-8 -*-
"""STAGEN 主求解器。

整合所有組件，執行完整的 3D 葉片幾何生成流程。
"""

from __future__ import annotations

from .blade_profile import BladeProfileGenerator
from .data_structures import StagenConfig
from .grid_generator import GridGenerator
from .io_handler import StagenFileHandler
from .projection import StreamSurfaceProjector


class StagenSolver:
    """STAGEN 主求解器。"""

    def __init__(self, config: StagenConfig) -> None:
        """初始化求解器。

        Args:
            config: STAGEN 配置
        """
        self.config = config
        self.profile_gen = BladeProfileGenerator()
        self.projector = StreamSurfaceProjector()
        self.grid_gen = GridGenerator()

    def solve(self) -> None:
        """執行完整求解流程。

        流程：
        1. 為每個葉片排生成幾何
        2. 為每個截面生成 2D 葉片
        3. 創建流線表面
        4. 投影到 3D
        5. 應用堆疊變換
        6. 生成網格
        """
        # TODO: 實現完整求解流程
        pass
```

### Priority 2: 測試和文檔
- 創建 `tests/test_stagen_solver.py`
- 更新 `README.md`
- 更新 `stagen/__init__.py` 導出

### Priority 3: 端到端測試
- 創建完整的輸入文件示例
- 測試從輸入到輸出的完整流程

## 📚 有用的資源

### 關鍵文件位置
```
multall_turbomachinery_design/
├── stagen/
│   ├── data_structures.py    # 數據結構
│   ├── blade_profile.py      # 2D 葉片生成
│   ├── projection.py          # 3D 投影
│   ├── grid_generator.py     # 網格生成
│   ├── io_handler.py          # I/O 處理
│   └── solver.py              # ← 待創建
├── examples/
│   ├── meangen_example.py     # MEANGEN 示例
│   └── stagen_example.py      # STAGEN 示例
└── tests/
    └── test_stagen_*.py       # STAGEN 測試
```

### 參考代碼
- MEANGEN 求解器: `meangen/mean_line_solver.py`
- STAGEN 示例: `examples/stagen_example.py`
- 原始 FORTRAN: `multall-open_20260120/STAGEN/stagen program/stagen-18.1.f`

## 🔍 檢查清單

開發完成前確認：
- [ ] 閱讀 `HANDOVER.md` 了解專案狀況
- [ ] 測試全部通過 (`pytest`)
- [ ] Ruff 檢查通過 (`ruff check . && ruff format --check .`)
- [ ] 更新 `README.md` 進度
- [ ] Commit message 符合規範
- [ ] **已推送到 GitHub** ← 最重要！

## 💡 開發建議

### 1. 先探索再開發
```python
# 使用 Read 工具讀取關鍵文件
Read("multall_turbomachinery_design/stagen/projection.py")
Read("examples/stagen_example.py")
```

### 2. 參考現有模式
- 查看 `meangen/mean_line_solver.py` 了解求解器結構
- 查看 `stagen/io_handler.py` 了解 I/O 模式
- 查看現有測試了解測試模式

### 3. 逐步開發
1. 先實現基本功能
2. 編寫測試
3. 確保測試通過
4. 提交並推送
5. 然後繼續下一個功能

### 4. 保持高質量
- 每個函數都要有 docstring（正體中文）
- 使用類型提示
- 錯誤處理要完善
- 測試覆蓋要充分

## 🚨 常見陷阱

### 1. 忘記推送到 GitHub
❌ **錯誤**: 開發完成但沒有 `git push`
✅ **正確**: 每次完成後立即推送

### 2. 使用錯誤的依賴安裝命令
❌ **錯誤**: `uv sync --all-extras`
✅ **正確**: `uv sync --extra dev`

### 3. 在錯誤的目錄工作
❌ **錯誤**: 在其他目錄運行命令
✅ **正確**: 確認在專案根目錄 (`pwd` 或 `cd`)

### 4. 測試沒過就提交
❌ **錯誤**: 測試失敗但仍然提交
✅ **正確**: 確保所有測試通過再提交

### 5. 忘記格式化代碼
❌ **錯誤**: 直接提交未格式化的代碼
✅ **正確**: 先 `ruff format .` 再提交

## 📞 需要幫助時

### 查看文檔
1. `HANDOVER.md` - 完整交接手冊
2. `README.md` - 專案 README
3. 現有測試文件 - 了解測試模式

### Git 操作
```bash
# 查看狀態
git status

# 查看最近提交
git log --oneline -10

# 查看分支
git branch

# 如果有問題，可以查看遠端狀態
git fetch
git status
```

## ✅ 確認理解

在開始開發前，請確認你理解了：
- [x] 要使用正體中文
- [x] 要自動繼續開發，不需詢問
- [x] 每次完成後**必須推送到 GitHub**
- [x] STAGEN 主求解器是首要任務
- [x] 測試必須全部通過
- [x] 代碼必須符合 Ruff 規範

---

**開始吧！繼續完成 STAGEN 主求解器。記得：先閱讀 HANDOVER.md，開發完成後推送到 GitHub！**

**重要**: 每次推送時記得加上 `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`
