# MULTALL 渦輪機械設計系統

[![CI](https://github.com/LostSunset/multall-turbomachinery-design/actions/workflows/ci.yml/badge.svg)](https://github.com/LostSunset/multall-turbomachinery-design/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/LostSunset/multall-turbomachinery-design.svg)](https://github.com/LostSunset/multall-turbomachinery-design/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/LostSunset/multall-turbomachinery-design.svg)](https://github.com/LostSunset/multall-turbomachinery-design/network)
[![GitHub issues](https://img.shields.io/github/issues/LostSunset/multall-turbomachinery-design.svg)](https://github.com/LostSunset/multall-turbomachinery-design/issues)

基於 Python 和 PySide6 的現代化渦輪機械設計系統，從原始 FORTRAN 77 程式碼移植而來。

## 📋 專案來源

本專案基於以下開源系統重新實現：
- **原始系統**: MULTALL Turbomachinery Design System
- **來源網站**: https://sites.google.com/view/multall-turbomachinery-design/to-download-the-system
- **原始語言**: FORTRAN 77
- **重構語言**: Python 3.14+ with PySide6

## ✨ 功能特點

### 核心模組

1. **MEANGEN** - 一維平均線設計
   - 速度三角形計算
   - 流道面積設計
   - 初始葉片數估算
   - 葉片輪廓猜測

2. **STAGEN** - 葉片幾何生成與操作
   - 葉片幾何生成
   - 網格細化與堆疊
   - 多級葉片組合
   - 網格間距控制

3. **MULTALL** - 三維 Navier-Stokes 求解器
   - 3D 流場求解
   - 軸流/混流/徑向流支援
   - 混合平面模型
   - 效率與性能預測
   - 逆向設計模式
   - 蒸汽渦輪專用查表功能

### 技術支援

- ✅ 軸流、混流、徑向流機械
- ✅ 多級設計
- ✅ 冷卻流道模擬
- ✅ 葉尖間隙效應
- ✅ 二次流模擬
- ✅ 激波捕捉
- ✅ 蒸汽性質查表
- ✅ 逆向設計

## 🚀 快速開始

### 系統需求

- Python 3.14 或更高版本
- [uv](https://github.com/astral-sh/uv) 套件管理器

### 安裝

```bash
# 克隆專案
git clone https://github.com/LostSunset/multall-turbomachinery-design.git
cd multall-turbomachinery-design

# 使用 uv 創建虛擬環境（Python 3.14，環境名 .venv314）
uv venv .venv314 --python 3.14

# 啟動虛擬環境
source .venv314/bin/activate  # Linux/Mac
# 或
.venv314\Scripts\activate     # Windows

# 安裝依賴
uv pip install -e ".[dev]"
```

### 執行

```bash
# 啟動圖形介面
python main.py

# 或執行特定模組
python -m multall_turbomachinery_design.meangen
python -m multall_turbomachinery_design.stagen
python -m multall_turbomachinery_design.multall
```

## 📁 專案結構

```
multall-turbomachinery-design/
├── multall_turbomachinery_design/    # 主程式包
│   ├── meangen/                      # 平均線設計模組
│   ├── stagen/                       # 葉片幾何模組
│   ├── multall/                      # 求解器模組
│   ├── ui/                           # PySide6 UI
│   └── utils/                        # 工具函數
├── tests/                            # 測試檔案
├── docs/                             # 文檔
├── multall-open_20260120/            # 原始 FORTRAN 程式碼（參考用）
├── pyproject.toml                    # 專案設定
└── README.md                         # 本檔案
```

## 🧪 測試

```bash
# 執行所有測試
pytest

# 執行特定測試並顯示涵蓋率
pytest tests/test_meangen.py --cov

# 執行測試並輸出中文（UTF-8 支援）
PYTHONIOENCODING=utf-8 pytest
```

## 🛠️ 開發

### 程式碼風格

本專案使用 [Ruff](https://github.com/astral-sh/ruff) 進行程式碼檢查和格式化。

```bash
# 檢查程式碼
ruff check .

# 自動修復
ruff check --fix .

# 格式化程式碼
ruff format .
```

### 類型檢查

```bash
mypy multall_turbomachinery_design/
```

## 🌍 國際化支援

- ✅ 完整 UTF-8 支援
- ✅ 正體中文介面
- ✅ 中文註釋與文檔
- ✅ 測試中文顯示正確

## 📊 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LostSunset/multall-turbomachinery-design&type=Date)](https://star-history.com/#LostSunset/multall-turbomachinery-design&Date)

## 🤝 貢獻

歡迎貢獻！請遵循以下步驟：

1. Fork 本專案
2. 創建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

## 📝 變更日誌

請參閱 [CHANGELOG.md](CHANGELOG.md) 了解各版本的詳細變更。

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案。

## 🙏 致謝

- 原始 MULTALL 系統的開發者
- [MULTALL Turbomachinery Design](https://sites.google.com/view/multall-turbomachinery-design/) 團隊

## 📧 聯絡

- GitHub Issues: [提交問題](https://github.com/LostSunset/multall-turbomachinery-design/issues)
- GitHub Discussions: [參與討論](https://github.com/LostSunset/multall-turbomachinery-design/discussions)

---

⭐ 如果這個專案對您有幫助，請給我們一顆星星！