# Claude Code 開發指南

本文件記錄使用 Claude Code 進行開發和自動發布的流程。

## 自動發布流程

### Release Agent

本專案包含一個名為 `release` 的使用者層級 agent，負責自動執行以下任務：

1. **執行測試** - 確保所有測試通過
2. **程式碼檢查** - 執行 Ruff 檢查並自動修復問題
3. **更新版本** - 根據變更類型自動更新版本號
4. **更新 CHANGELOG** - 記錄變更內容
5. **更新 README** - 如需要，更新徽章和資訊
6. **Git 提交** - 提交所有變更
7. **建立標籤** - 建立版本標籤
8. **推送到 GitHub** - 推送 commits 和 tags

### 版本規則

| 變更類型 | 版本變更 | 範例 |
|---------|---------|------|
| Bug 修復或效能提升 | PATCH | v0.2.1 → v0.2.2 |
| 新功能 | MINOR | v0.2.1 → v0.3.0 |
| 重大變更 | MAJOR | v0.2.1 → v1.0.0 |

### 使用方式

#### 1. 基本發布

```bash
# 完成開發工作後，使用 release agent
/release "新增 MEANGEN 速度三角形計算功能"
```

Agent 會自動：
- 判斷這是新功能（MINOR 版本更新）
- 執行所有測試
- 更新版本號從 0.1.0 → 0.2.0
- 更新 CHANGELOG.md
- 提交並推送

#### 2. 指定版本類型

```bash
# Bug 修復（PATCH）
/release "修復 STAGEN 網格生成的記憶體洩漏問題" --type patch

# 新功能（MINOR，預設）
/release "新增 MULTALL 逆向設計模式" --type minor

# 重大變更（MAJOR）
/release "重構整個 API，不向後相容" --type major
```

#### 3. 跳過測試（僅文檔更新）

```bash
# 僅更新文檔時可跳過測試
/release "更新 README 安裝說明" --skip-tests
```

### 手動發布流程

如果需要手動發布，請遵循以下步驟：

```bash
# 1. 確保所有變更已提交
git status

# 2. 執行測試
uv run pytest

# 3. 執行程式碼檢查
uv run ruff check --fix .
uv run ruff format .

# 4. 更新版本號（編輯以下檔案）
# - pyproject.toml
# - multall_turbomachinery_design/__init__.py
# - CHANGELOG.md

# 5. 提交變更
git add .
git commit -m "Release v0.x.y"

# 6. 建立標籤
git tag -a v0.x.y -m "Release v0.x.y"

# 7. 推送到 GitHub
git push origin main
git push origin v0.x.y
```

## 開發工作流程

### 1. 開始新功能

```bash
# 建立功能分支
git checkout -b feature/new-feature

# 啟動虛擬環境
source .venv314/bin/activate  # Linux/Mac
# 或
.venv314\Scripts\activate     # Windows

# 開始開發...
```

### 2. 測試驅動開發

```bash
# 先寫測試
# tests/test_new_feature.py

# 執行測試（應該失敗）
uv run pytest tests/test_new_feature.py

# 實作功能

# 再次執行測試（應該通過）
uv run pytest tests/test_new_feature.py

# 執行所有測試
uv run pytest
```

### 3. 程式碼品質檢查

```bash
# 自動修復程式碼風格問題
uv run ruff check --fix .

# 格式化程式碼
uv run ruff format .

# 類型檢查
uv run mypy multall_turbomachinery_design/
```

### 4. 提交變更

```bash
# 檢查狀態
git status

# 加入變更
git add .

# 提交（使用有意義的訊息）
git commit -m "feat: 新增 MEANGEN 速度三角形計算

- 實作速度三角形計算演算法
- 新增單元測試
- 更新文檔
"
```

### 5. 推送和發布

```bash
# 推送到 GitHub
git push origin feature/new-feature

# 建立 Pull Request（在 GitHub 網頁上）

# 合併後，在 main 分支使用 release agent
git checkout main
git pull
/release "新增 MEANGEN 速度三角形計算功能"
```

## 常見問題

### Q: Ruff 檢查失敗怎麼辦？

A: 大多數問題可以自動修復：

```bash
# 自動修復
uv run ruff check --fix .

# 格式化
uv run ruff format .
```

常見錯誤：
- **W293** (空白行包含空格) → 自動修復
- **E722** (bare except) → 改為 `except Exception:`
- **I001** (import 未排序) → 自動修復
- **F401** (未使用的 import) → 刪除或加 `# noqa: F401`

### Q: 如何處理 forward reference 錯誤？

A: 在檔案開頭加入：

```python
from __future__ import annotations
```

### Q: 測試中文顯示亂碼？

A: 確保設置環境變數：

```bash
export PYTHONIOENCODING=utf-8  # Linux/Mac
set PYTHONIOENCODING=utf-8     # Windows
```

### Q: 如何跳過某次 CI 檢查？

A: 在 commit 訊息中加入 `[skip ci]`：

```bash
git commit -m "docs: 更新 README [skip ci]"
```

## UTF-8 / 正體中文支援檢查清單

- ✅ 所有 `.py` 檔案開頭有 `# -*- coding: utf-8 -*-`
- ✅ 所有檔案使用 UTF-8 編碼儲存
- ✅ 檔案讀寫明確指定 `encoding="utf-8"`
- ✅ 測試執行時設定 `PYTHONIOENCODING=utf-8`
- ✅ CI 設定正確的 locale 環境變數
- ✅ Pytest assertion 可正確顯示中文

## 提交訊息規範

使用 Conventional Commits 格式：

```
<類型>(<範圍>): <簡短描述>

<詳細描述>

<註腳>
```

### 類型

- `feat`: 新功能
- `fix`: Bug 修復
- `docs`: 文檔更新
- `style`: 程式碼格式（不影響功能）
- `refactor`: 重構
- `perf`: 效能改進
- `test`: 測試相關
- `chore`: 建置工具或輔助工具變動

### 範例

```bash
# 新功能
git commit -m "feat(meangen): 新增速度三角形計算功能"

# Bug 修復
git commit -m "fix(stagen): 修復網格生成記憶體洩漏"

# 文檔
git commit -m "docs: 更新安裝說明"

# 重大變更
git commit -m "feat(api)!: 重構 API 介面

BREAKING CHANGE: API 不向後相容"
```

## 開發環境設置

### 首次設置

```bash
# 克隆專案
git clone https://github.com/LostSunset/multall-turbomachinery-design.git
cd multall-turbomachinery-design

# 建立虛擬環境
uv venv .venv314 --python 3.14

# 啟動虛擬環境
source .venv314/bin/activate  # Linux/Mac
.venv314\Scripts\activate     # Windows

# 安裝依賴（包含開發依賴）
uv pip install -e ".[dev]"

# 驗證安裝
python -c "import multall_turbomachinery_design; print(multall_turbomachinery_design.__version__)"
```

### VS Code 設置

建議的 `.vscode/settings.json`：

```json
{
  "python.defaultInterpreterPath": ".venv314/bin/python",
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "files.encoding": "utf8",
  "files.autoGuessEncoding": false
}
```

## 相關資源

- [原始 MULTALL 系統](https://sites.google.com/view/multall-turbomachinery-design/)
- [uv 文檔](https://github.com/astral-sh/uv)
- [PySide6 文檔](https://doc.qt.io/qtforpython/)
- [Ruff 文檔](https://docs.astral.sh/ruff/)
- [Pytest 文檔](https://docs.pytest.org/)

---

最後更新: 2025-01-29
