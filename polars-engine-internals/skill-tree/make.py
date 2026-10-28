#!/usr/bin/env python3
"""建置技能樹並產生 GitHub Pages 用的獨立版。

用法:  python3 make.py

1. 執行 build.py（驗證不過會 exit 1，不要繞過去手改產物）
2. 把片段版 polars-engine.html 包成可直接開啟的 index.html
3. 刪掉除錯副產物 skill-tree.json
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
FRAGMENT = HERE / "polars-engine.html"


def standalone(src: pathlib.Path, out: pathlib.Path) -> None:
    """以第一個 </style> 為界切開片段：前半放進 head、後半放進 body。"""
    text = src.read_text(encoding="utf-8")
    marker = "</style>"
    cut = text.find(marker)
    if cut == -1:
        sys.exit(f"❌ {src.name} 找不到 </style>")
    cut += len(marker)
    out.write_text(
        '<!DOCTYPE html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n'
        f"{text[:cut]}\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>{text[cut:]}</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"✅ 獨立版 → {out.relative_to(HERE.parent)}  ({out.stat().st_size // 1024} KB)")


def main() -> int:
    code = subprocess.run([sys.executable, "build.py"], cwd=HERE).returncode
    if code:
        return code
    standalone(FRAGMENT, HERE / "index.html")
    (HERE / "skill-tree.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
