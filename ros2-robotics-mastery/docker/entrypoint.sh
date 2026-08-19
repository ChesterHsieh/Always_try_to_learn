#!/usr/bin/env bash
set -e

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
