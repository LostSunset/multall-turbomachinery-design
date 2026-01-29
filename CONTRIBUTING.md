# 貢獻指南

感謝您對 MULTALL 渦輪機械設計系統的興趣！我們歡迎所有形式的貢獻。

## 行為準則

- 尊重所有貢獻者
- 使用友善和包容的語言
- 接受建設性的批評
- 專注於對社群最有利的事情

## 如何貢獻

### 回報問題

如果您發現 bug 或有功能建議，請：

1. 檢查 [Issues](https://github.com/LostSunset/multall-turbomachinery-design/issues) 確認問題尚未被回報
2. 建立新的 Issue，提供詳細資訊：
   - Bug: 重現步驟、預期行為、實際行為、環境資訊
   - 功能建議: 使用情境、預期效果、可能的實現方式

### 提交程式碼

1. **Fork 專案**
   ```bash
   # 在 GitHub 上 fork 專案
   git clone https://github.com/你的使用者名稱/multall-turbomachinery-design.git
   cd multall-turbomachinery-design
   ```

2. **設置開發環境**
   ```bash
   # 建立虛擬環境
   uv venv .venv314 --python 3.14
   source .venv314/bin/activate  # Linux/Mac

   # 安裝依賴
   uv pip install -e ".[dev]"
   ```

3. **建立分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

4. **撰寫程式碼**
   - 遵循專案的程式碼風格（使用 Ruff）
   - 為新功能編寫測試
   - 確保所有測試通過
   - 更新相關文檔

5. **程式碼品質檢查**
   ```bash
   # 執行測試
   PYTHONIOENCODING=utf-8 pytest

   # 程式碼檢查和自動修復
   ruff check --fix .
   ruff format .

   # 類型檢查
   mypy multall_turbomachinery_design/
   ```

6. **提交變更**
   ```bash
   git add .
   git commit -m "feat: 簡短描述你的變更"
   ```

   使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：
   - `feat:` 新功能
   - `fix:` Bug 修復
   - `docs:` 文檔更新
   - `style:` 程式碼格式（不影響功能）
   - `refactor:` 重構
   - `test:` 測試相關
   - `chore:` 建置或輔助工具變動

7. **推送到您的 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **建立 Pull Request**
   - 前往 GitHub 建立 Pull Request
   - 填寫 PR 描述，說明變更內容和原因
   - 連結相關的 Issue
   - 等待 code review

## 程式碼風格

### Python 風格指南

- 遵循 PEP 8
- 使用 Ruff 進行程式碼檢查和格式化
- 行長度限制: 100 字元
- 使用類型提示（Type Hints）

### 檔案編碼

- 所有檔案使用 UTF-8 編碼
- 所有 Python 檔案開頭包含 `# -*- coding: utf-8 -*-`
- 確保正體中文顯示正確

### 文檔字串

使用 Google 風格的 docstring：

```python
def function_name(param1: int, param2: str) -> bool:
    """簡短描述函數功能。

    詳細描述函數的行為、用途和注意事項。

    Args:
        param1: 參數1的描述
        param2: 參數2的描述

    Returns:
        返回值的描述

    Raises:
        ValueError: 在什麼情況下會拋出此異常
    """
    pass
```

### 測試

- 為新功能編寫單元測試
- 測試檔案放在 `tests/` 目錄
- 測試檔案名稱: `test_*.py`
- 測試函數名稱: `test_*`
- 使用 pytest fixtures 共享測試資料
- 目標測試覆蓋率: > 80%

## 開發流程

### 分支策略

- `main`: 穩定的生產分支
- `feature/*`: 新功能開發
- `fix/*`: Bug 修復
- `docs/*`: 文檔更新

### 版本號規則

遵循 [語義化版本](https://semver.org/lang/zh-TW/)：

- `MAJOR.MINOR.PATCH`
- MAJOR: 重大變更，不向後相容
- MINOR: 新功能，向後相容
- PATCH: Bug 修復，向後相容

### CI/CD

所有 Pull Request 會自動執行：
- 程式碼風格檢查（Ruff）
- 單元測試（pytest）
- 類型檢查（mypy）
- 多平台測試（Linux, Windows, macOS）

## 需要幫助的領域

目前特別需要貢獻的領域：

1. **FORTRAN 到 Python 轉換**
   - MEANGEN 模組實現
   - STAGEN 模組實現
   - MULTALL 求解器實現

2. **UI 開發**
   - 使用 PySide6 開發圖形介面
   - 改善使用者體驗
   - 視覺化功能

3. **測試**
   - 增加測試覆蓋率
   - 編寫整合測試
   - 效能測試

4. **文檔**
   - 使用指南
   - API 文檔
   - 教學範例
   - 正體中文翻譯

5. **效能優化**
   - 數值計算優化
   - 記憶體使用優化
   - 平行處理

## 問題與討論

- 技術問題: 開啟 [Issue](https://github.com/LostSunset/multall-turbomachinery-design/issues)
- 功能討論: 使用 [Discussions](https://github.com/LostSunset/multall-turbomachinery-design/discussions)
- 安全問題: 請私下聯繫維護者

## 授權

提交貢獻即表示您同意您的貢獻將以 MIT 授權發布。

---

感謝您的貢獻！ 🎉
