#!/usr/bin/env bash
set -e

# build/install/log 是 docker named volume，首次建立時屬於 root，
# 非 root 的 dev 使用者會無法寫入而導致 colcon build 失敗。
# 每次啟動都確認一次擁有者，讓 volume 被刪掉重建後也能自動修好。
for d in /ros2_ws/build /ros2_ws/install /ros2_ws/log; do
  if [ -d "$d" ] && [ ! -w "$d" ]; then
    sudo chown "$(id -u):$(id -g)" "$d" 2>/dev/null || true
  fi
done

source /opt/ros/jazzy/setup.bash
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash

# 若有安裝 VNC 且 DISPLAY 指向容器內，就把虛擬桌面拉起來
if [ "${DISPLAY}" = ":99" ] && command -v Xvfb >/dev/null 2>&1; then
  if ! pgrep -x Xvfb >/dev/null; then
    Xvfb :99 -screen 0 1920x1080x24 +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 1
    fluxbox >/dev/null 2>&1 &
    x11vnc -display :99 -forever -shared -nopw -quiet >/dev/null 2>&1 &
    websockify --web=/usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &
    echo "🖥  noVNC 已啟動 → 瀏覽器開 http://localhost:6080/vnc.html"
  fi
fi

exec "$@"
