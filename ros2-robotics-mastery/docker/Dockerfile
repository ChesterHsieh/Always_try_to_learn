# ROS 2 Jazzy 開發環境 — Apple Silicon (linux/arm64) 原生
#
# build:  docker compose build
# run:    docker compose up -d && docker compose exec ros2 bash
FROM ros:jazzy-ros-base

ARG WITH_VNC=1
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ── 基礎工具 ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl wget vim less sudo \
      python3-pip python3-venv python3-dev \
      python3-colcon-common-extensions python3-rosdep python3-vcstool \
      ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# ── ROS 2 桌面套件 + 本專案主要用到的堆疊 ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-desktop \
      ros-jazzy-rviz2 \
      ros-jazzy-rqt* \
      ros-jazzy-foxglove-bridge \
      ros-jazzy-xacro \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-tf2-tools ros-jazzy-tf-transformations \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# ── Gazebo Harmonic（Jazzy 的官方配對，非 Gazebo Classic）──
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# ── MoveIt 2（第七章用）─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-moveit \
      ros-jazzy-moveit-visual-tools \
    && rm -rf /var/lib/apt/lists/* || echo "moveit 安裝失敗，第七章再處理"

# ── Nav2（第六章支線，可選）────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
    && rm -rf /var/lib/apt/lists/* || echo "nav2 安裝失敗，第六章再處理"

# ── VNC（跑 Gazebo / RViz 建議開啟）──────────────────────
RUN if [ "$WITH_VNC" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify fluxbox mesa-utils libgl1-mesa-dri \
      && rm -rf /var/lib/apt/lists/* ; \
    fi

# ── Python 工具（分析 rosbag 用得到）─────────────────────
# 注意：open3d-cpu 在 linux/arm64 沒有 wheel，刻意不裝。
# 需要點雲處理時改用容器內的 PCL，或在 x86 機器上另外處理。
RUN pip3 install --break-system-packages --no-cache-dir \
      numpy pandas matplotlib pyarrow \
      mcap mcap-ros2-support

# ── 非 root 使用者 ────────────────────────────────────────
# Ubuntu Noble 的 base image 已內建 UID/GID 1000 的 ubuntu 帳號。
# 若目標 UID 已被占用就把既有帳號改名，否則才新建，
# 這樣掛載進來的檔案擁有者才會對得上 macOS 端的 501:20 對映。
RUN set -eux; \
    if getent group "$GID" >/dev/null; then \
      existing_group="$(getent group "$GID" | cut -d: -f1)"; \
      [ "$existing_group" = "$USERNAME" ] || groupmod -n "$USERNAME" "$existing_group"; \
    else \
      groupadd -g "$GID" "$USERNAME"; \
    fi; \
    if getent passwd "$UID" >/dev/null; then \
      existing_user="$(getent passwd "$UID" | cut -d: -f1)"; \
      if [ "$existing_user" != "$USERNAME" ]; then \
        usermod -l "$USERNAME" -d "/home/$USERNAME" -m "$existing_user"; \
      fi; \
      usermod -g "$GID" -s /bin/bash "$USERNAME"; \
    else \
      useradd -m -u "$UID" -g "$GID" -s /bin/bash "$USERNAME"; \
    fi; \
    mkdir -p "/home/$USERNAME"; \
    chown "$UID:$GID" "/home/$USERNAME"; \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"; \
    chmod 0440 "/etc/sudoers.d/$USERNAME"

RUN mkdir -p /ros2_ws/src && chown -R $UID:$GID /ros2_ws

COPY entrypoint.sh /entrypoint.sh
# 用 0755 而非 chmod +x：build context 來自 macOS 時可能帶著 711 之類的模式，
# 非 root 使用者會因為沒有讀取權限而無法執行 script。
RUN chmod 0755 /entrypoint.sh

USER $USERNAME
WORKDIR /ros2_ws

RUN echo "source /opt/ros/jazzy/setup.bash"                        >> ~/.bashrc && \
    echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-42}"              >> ~/.bashrc && \
    echo "export RCUTILS_COLORIZED_OUTPUT=1"                       >> ~/.bashrc && \
    echo "export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] [{time}] [{name}]: {message}'" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
