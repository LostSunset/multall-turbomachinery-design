# -*- coding: utf-8 -*-
"""MULTALL 渦輪機械設計系統主程式。

啟動圖形使用者介面的主要進入點。
"""

from __future__ import annotations

import sys

from multall_turbomachinery_design.ui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
