# MULTALL 渦輪機械設計系統 - 交接手冊

## 📋 專案概述

基於 Python 3.14 和 PySide6 的現代化渦輪機械設計系統，從 FORTRAN 77 移植而來。

**GitHub 倉庫**: https://github.com/LostSunset/multall-turbomachinery-design

## 🎯 開發目標

**最終目標**: 輸出完整 3D 葉片 CAD，包含流道（shroud、hub）、rotors、stators

**開發原則**:
- 自動持續開發，不需詢問用戶
- 每次完成開發後必須推送到 GitHub
- 使用正體中文進行所有溝通和文檔

## 📊 當前進度

### MEANGEN - 平均線設計模組 ✅ 100% 完成
- ✅ 數據結構定義
- ✅ 氣體性質計算
- ✅ 速度三角形計算
- ✅ 流表面生成
- ✅ 葉片幾何生成
- ✅ I/O 處理器
- ✅ 主求解器
- ✅ 使用示例 (`examples/meangen_example.py`)

### STAGEN - 葉片幾何模組 🚧 約 85% 完成
- ✅ 數據結構定義 (`stagen/data_structures.py`)
- ✅ 2D 葉片截面生成器 (`stagen/blade_profile.py`, 100% 覆蓋率)
- ✅ 3D 流線投影器 (`stagen/projection.py`, 99% 覆蓋率)
- ✅ 網格生成器 (`stagen/grid_generator.py`, 99% 覆蓋率)
- ✅ I/O 處理器 (`stagen/io_handler.py`, 96% 覆蓋率)
- ✅ 使用示例 (`examples/stagen_example.py`)
- ⏳ **主求解器** (待完成，最後一個核心組件)
- ⏳ CAD 輸出 (待 CadQuery 支援 Python 3.14)

### MULTALL - 3D 求解器 🔮 規劃中
- ⏳ Navier-Stokes 求解器
- ⏳ 混合平面模型
- ⏳ 逆向設計
- ⏳ 後處理工具

### 測試與覆蓋率
- ✅ **113 個測試**，全部通過
- ✅ **94% 程式碼覆蓋率**
- ✅ CI/CD 自動化

## 🛠️ 開發環境設置

### Windows 環境

```powershell
# 克隆專案
git clone https://github.com/LostSunset/multall-turbomachinery-design.git
cd multall-turbomachinery-design

# 使用 uv 創建虛擬環境（Python 3.14）
uv venv .venv --python 3.14

# 啟動虛擬環境
.venv\Scripts\activate

# 安裝開發依賴（注意：不要使用 --all-extras，CAD 依賴暫時不可用）
uv pip install -e ".[dev]"

# 或使用 uv sync（推薦）
uv sync --extra dev
```

### Linux/WSL 環境

```bash
# 啟動虛擬環境
source .venv/bin/activate

# 安裝開發依賴
uv pip install -e ".[dev]"
```

### 驗證環境

```bash
# 運行測試
pytest

# 檢查代碼風格
ruff check .

# 格式化代碼
ruff format .
```

## 📝 開發工作流程

### 1. 開始開發前

```bash
# 拉取最新代碼
git pull origin main

# 確認當前分支
git branch

# 確認工作目錄乾淨
git status
```

### 2. 開發過程中

- 使用 Python 3.14 語法
- 所有註釋和文檔使用正體中文
- 遵循現有代碼風格
- 為新功能編寫測試
- 確保測試通過且覆蓋率不降低

### 3. 完成開發後

```bash
# 運行測試
pytest tests/ --cov

# 檢查並修復代碼風格
ruff check . --fix --unsafe-fixes
ruff format .

# 添加文件
git add <files>

# 提交（使用規範的 commit message）
git commit -m "類型(範圍): 簡短描述

詳細描述...

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 推送到 GitHub
git push origin main
```

## 📐 Commit Message 規範

### 類型 (Type)
- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 代碼風格（不影響功能）
- `refactor`: 重構
- `test`: 測試相關
- `chore`: 構建/工具相關

### 範圍 (Scope)
- `meangen`: MEANGEN 模組
- `stagen`: STAGEN 模組
- `multall`: MULTALL 模組
- `deps`: 依賴相關
- `ci`: CI/CD 相關

### 示例

```
feat(stagen): 實現主求解器與完整測試

新增 stagen/solver.py：
- StagenSolver 類整合所有組件
- 完整的 3D 葉片幾何生成流程
- 支援多截面、多葉片排

測試：
- 新增 15 個測試，全部通過
- 覆蓋率達到 98%

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## 🔧 關鍵技術細節

### STAGEN 架構

```
用戶輸入 (stagen.dat)
    ↓
StagenInputReader (讀取配置)
    ↓
StagenSolver (主求解器) ← 待實現
    ├→ BladeProfileGenerator (2D 葉片生成)
    ├→ StreamSurfaceProjector (3D 投影)
    ├→ GridGenerator (網格生成)
    └→ StagenOutputWriter (輸出結果)
    ↓
輸出文件 (stage_old.dat, stage_new.dat, stagen.out)
```

### 主要數據流

1. **輸入階段**: 讀取 `stagen.dat` → `StagenConfig`
2. **生成階段**:
   - 為每個截面生成 2D 葉片 → `BladeProfile2D`
   - 創建流線表面 → `StreamSurface3D`
   - 投影到 3D → `BladeSection3D`
   - 應用堆疊變換
   - 生成網格
3. **輸出階段**: 寫入多種格式的輸出文件

### 下一步開發重點

**STAGEN 主求解器** (`stagen/solver.py`):
```python
class StagenSolver:
    """STAGEN 主求解器，整合所有組件。"""

    def __init__(self, config: StagenConfig):
        self.config = config
        self.profile_gen = BladeProfileGenerator()
        self.projector = StreamSurfaceProjector()
        self.grid_gen = GridGenerator()

    def solve(self) -> None:
        """執行完整求解流程。"""
        # 1. 為每個葉片排生成幾何
        for row in self.config.blade_rows:
            # 2. 為每個截面生成 2D 葉片
            for section_num in range(self.config.nosect):
                # 3. 生成 2D 葉片截面
                # 4. 創建流線表面
                # 5. 投影到 3D
                # 6. 應用堆疊變換
            # 7. 生成網格
```

## 🚨 已知問題

### CadQuery 依賴問題
- **問題**: CadQuery 尚未支援 Python 3.14
- **狀態**: 已在 `pyproject.toml` 中註釋掉
- **解決方案**: 待 CadQuery 發布 Python 3.14 支援後取消註釋
- **影響**: CAD 輸出功能暫時不可用，不影響核心功能

### 使用 uv sync 注意事項
```bash
# ✅ 正確：只安裝開發依賴
uv sync --extra dev

# ❌ 錯誤：會嘗試安裝 CAD 依賴並失敗
uv sync --all-extras
```

## 📚 重要文件說明

### 配置文件
- `pyproject.toml`: 專案配置、依賴、工具設置
- `.github/workflows/ci.yml`: CI/CD 配置

### 示例文件
- `examples/meangen_example.py`: MEANGEN 使用示例（4 個示例）
- `examples/stagen_example.py`: STAGEN 使用示例（4 個示例）

### 測試文件結構
```
tests/
├── test_import.py                    # 導入測試
├── test_meangen_*.py                 # MEANGEN 測試 (46 個)
└── test_stagen_*.py                  # STAGEN 測試 (67 個)
    ├── test_stagen_data_structures.py
    ├── test_stagen_blade_profile.py
    ├── test_stagen_projection.py
    ├── test_stagen_grid.py
    └── test_stagen_io.py
```

## 🎓 學習資源

### 原始 FORTRAN 代碼
- 位置: `multall-open_20260120/`
- STAGEN: `STAGEN/stagen program/stagen-18.1.f`
- MEANGEN: `MEANGEN/meangen program/meangen-17.3.f`

### 參考網站
- 原始系統: https://sites.google.com/view/multall-turbomachinery-design/

## 🔐 Git 操作備忘

```bash
# 查看當前狀態
git status

# 查看最近提交
git log --oneline -10

# 查看差異
git diff

# 暫存特定文件
git add <file>

# 暫存所有修改
git add -A

# 提交
git commit -m "message"

# 推送
git push origin main

# 拉取最新更新
git pull origin main
```

## ✅ 開發檢查清單

每次開發完成前：

- [ ] 所有測試通過 (`pytest`)
- [ ] 代碼覆蓋率不降低
- [ ] Ruff 檢查通過 (`ruff check .`)
- [ ] 代碼已格式化 (`ruff format .`)
- [ ] 更新相關文檔 (README.md)
- [ ] Commit message 符合規範
- [ ] 已推送到 GitHub

## 📞 聯絡資訊

- **GitHub Issues**: https://github.com/LostSunset/multall-turbomachinery-design/issues
- **GitHub Discussions**: https://github.com/LostSunset/multall-turbomachinery-design/discussions

## 🎯 下一階段目標

### 立即要做 (Priority 1)
1. **完成 STAGEN 主求解器** (`stagen/solver.py`)
   - 整合所有已完成的組件
   - 實現完整的 3D 葉片生成流程
   - 編寫完整測試
   - 創建端到端示例

### 近期計劃 (Priority 2)
2. **STAGEN 端到端測試**
   - 創建完整的輸入文件示例
   - 測試從輸入到輸出的完整流程
   - 驗證輸出文件格式

3. **文檔完善**
   - 完整的 API 文檔
   - 使用者指南
   - 理論背景說明

### 中期計劃 (Priority 3)
4. **UI 整合**
   - 整合 PySide6 圖形介面
   - 視覺化結果展示

5. **CAD 輸出**
   - 待 CadQuery 支援 Python 3.14
   - 實現 3D 模型輸出

### 長期計劃 (Priority 4)
6. **MULTALL 求解器**
   - 3D Navier-Stokes 求解器
   - 性能預測
   - 逆向設計

---

**最後更新**: 2026-01-29
**當前版本**: v0.1.0
**Python 版本**: 3.14.2
**測試狀態**: 113/113 通過
**覆蓋率**: 94%
