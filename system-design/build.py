#!/usr/bin/env python3
"""system-design 技能樹建置器。

用法:  python3 build.py <題目資料夾>       例如  python3 build.py url-shortener

各題目資料夾只維護 tree_data.py（單一資料來源）；產生器與樣板集中在 _tooling/，
建置時暫時複製進題目資料夾，產生 <slug>.html 後移除，避免每個題目各自複製一份工具。
驗證不過會 exit 1 —— 不要繞過去手改產物 HTML。
"""
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
TOOLING = HERE / "_tooling"
TOOLS = ("build.py", "index.template.html")


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
        return subprocess.run([sys.executable, "build.py"], cwd=topic).returncode
    finally:
        for path in copied:
            path.unlink(missing_ok=True)
        (topic / "skill-tree.json").unlink(missing_ok=True)  # 除錯副產物，不入庫


if __name__ == "__main__":
    sys.exit(main())
