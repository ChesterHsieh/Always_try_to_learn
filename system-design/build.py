#!/usr/bin/env python3
"""system-design 技能樹建置器。

用法:  python3 build.py <題目資料夾>       例如  python3 build.py url-shortener

各題目資料夾只維護 tree_data.py（單一資料來源）；產生器與樣板集中在 _tooling/，
建置時暫時複製進題目資料夾，產生 <slug>.html 後移除，避免每個題目各自複製一份工具。
驗證不過會 exit 1 —— 不要繞過去手改產物 HTML。

產出兩份：
  <slug>.html  片段版，發布成 Artifact 用（宿主會自行包覆 doctype/html/body）
  index.html   獨立版，加上 doctype 與 viewport，供 GitHub Pages 或直接用瀏覽器開啟
"""
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
TOOLING = HERE / "_tooling"
TOOLS = ("build.py", "index.template.html")


def standalone(topic: pathlib.Path) -> None:
    """把片段版包成可直接開啟的 index.html。

    片段的開頭是 meta / title / style，其餘是內容；以第一個 </style> 為界切開，
    前半放進 head、後半放進 body，並補上 viewport 讓行動裝置正確縮放。
    """
    skip = {"index.html", *TOOLS}   # 建置中暫存的樣板還在，要排除
    fragments = [p for p in topic.glob("*.html") if p.name not in skip]
    if len(fragments) != 1:
        print(f"⚠️  {topic.name} 有 {len(fragments)} 份片段 HTML，跳過獨立版產生")
        return
    src = fragments[0].read_text(encoding="utf-8")
    marker = "</style>"
    cut = src.find(marker)
    if cut == -1:
        print(f"⚠️  {fragments[0].name} 找不到 </style>，跳過獨立版產生")
        return
    cut += len(marker)
    head, body = src[:cut], src[cut:]
    out = topic / "index.html"
    out.write_text(
        '<!DOCTYPE html>\n<html lang="zh-Hant">\n<head>\n'
        f"{head}\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>{body}</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"✅ 獨立版 → {out.relative_to(topic.parent)}  ({out.stat().st_size // 1024} KB)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    topic = HERE / sys.argv[1]
    if not (topic / "tree_data.py").exists():
        print(f"❌ 找不到 {topic}/tree_data.py")
        return 1

    copied = []
    try:
        for name in TOOLS:
            dst = topic / name
            if dst.exists():
                print(f"❌ {dst} 已存在，請先移除以免覆蓋")
                return 1
            shutil.copy2(TOOLING / name, dst)
            copied.append(dst)
        code = subprocess.run([sys.executable, "build.py"], cwd=topic).returncode
        if code == 0:
            standalone(topic)
        return code
    finally:
        for path in copied:
            path.unlink(missing_ok=True)
        (topic / "skill-tree.json").unlink(missing_ok=True)  # 除錯副產物，不入庫


if __name__ == "__main__":
    sys.exit(main())
