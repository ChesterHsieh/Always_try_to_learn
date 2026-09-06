# NOTES —— 「我以為 X，結果是 Y」

> 每次踩雷、每次領悟都記一行。這是之後寫技術文件與面試故事的素材。

## 2026-09-03 · 專案立案

- 我以為單鏡頭做不到 3D，結果 SwingVision 就是單鏡頭：靠球場幾何（homography）+ 彈道模型，不靠深度。
- 我以為 SwingVision 有 kitchen 違例偵測，結果官方完全沒有這功能。這代表它不在「基本功能」清單裡，我也不做。
- 我以為計分要手動輸入，結果決定只做全自動：錯誤率變成一個可量化、可展示的數據，而不是要藏起來的東西。

## 2026-09-03 · 第一段素材

- 我以為轉播影片鏡頭會一直切、不能用，結果 PPA 的主機位是固定廣角，切鏡剪掉後剩 92% 可用。
- 我以為 AI 計分的 ground truth 要人工記，結果轉播比分板就在畫面右上角，OCR 就好。
- 這段只到 8-10 沒打完，驗收用的整局還是要另找。

## 2026-09-06 · Phase 0 C++ 骨架與工具鏈

### 工具鏈
- 我以為要在 Mac 上裝 C++ 工具鏈，結果不用：ROS 2 的 C++ 開發全部在容器內（GCC 13.3 + CMake 3.28 + OpenCV 4.6）。Mac 上的 Apple clang 21 完全用不到，因為 macOS 上沒有 ROS 2 二進位套件。編輯在 Mac、編譯在容器，`ros2_ws/src` 用 bind mount 共用。
- 我以為 `docker` 指令就是 Docker Desktop，結果這台裝的是 **Rancher Desktop**（版本號 `29.5.3-rd`，socket 指向 `~/.rd/docker.sock`）。要開的是 Rancher Desktop 不是 Docker Desktop。
- `docker compose exec` **不會**經過 entrypoint，`.bashrc` 又對非互動 shell 提早 return，所以每個 exec 都要自己 `source /opt/ros/jazzy/setup.bash`，否則 `ament_cmake` 找不到。

### Dockerfile 的三個坑（都已修）
- 我以為 `useradd` 的守衛 `if ! id -u $UID` 是安全的，結果 Ubuntu Noble 的 base image **已內建 UID/GID 1000 的 `ubuntu` 帳號**，守衛直接跳過建立，後面 `USER dev` 就 `unable to find user dev`。改成偵測到 UID 被占用時 `usermod -l` 改名。
- `open3d-cpu` 在 linux/arm64 **沒有 wheel**，而原本的 `||` fallback 寫法在 `pip` 回傳非零時仍會讓整層失敗。直接移除。
- `COPY` 進來的 `entrypoint.sh` 帶著 macOS 端的 `711` 權限，`chmod +x` 不會補上 read bit，非 root 使用者執行時得到 `Permission denied`（bash 執行 script 需要**讀取**權限，不只是 execute）。改成 `chmod 0755`。
- `build/`、`install/`、`log/` 是 named volume，首次建立屬於 root，`dev` 寫不進去。已在 entrypoint 加入自動 chown，volume 被刪掉重建也能自動修好。

### 介面設計
- msg 欄位**不能叫 `auto`**——那是 C/C++ 保留字，rosidl 產生的 C struct 會編不過（`bool auto;`）。已改名 `auto_detect`。這類保留字問題在 Python-only 的專案不會遇到。
- `cv_bridge` 的標頭 Jazzy 改名為 `.hpp`，Humble 只有 `.h`。用 `#if __has_include(<cv_bridge/cv_bridge.hpp>)` 讓同一份程式碼兩邊都能編，符合雙版本要求。
- `MatchState` 原本設計內嵌 `Rally[] point_history`，而 `Rally` 又內嵌 `ShotEvent[]`，訊息大小會隨比賽平方成長。改成只帶 `uint32[] point_history`（rally_id），完整內容從 `/rally` topic 或 rosbag 取。

### Kalman 濾波的真實 bug（測試抓到的）
- 我以為手寫 `P = F P Fᵀ` 很簡單，結果第一版**寫壞了對稱性**：第二個迴圈原地讀寫同一列，同列中先被覆寫的元素污染了後面的計算，導致 `P[px][vx]` 有值但 `P[vx][px]` 恆為 0。
- 症狀非常有欺騙性：位置估計看起來「差不多對」（會朝觀測移動），但**速度永遠是 0.00**，於是估計逐格落後真值，最後被離群值閘門擋下。如果只用眼睛看影片上的軌跡，很可能會誤判成「參數沒調好」而去亂調 process noise。
- 這就是第 8 節補償條款要的東西：`ConvergesToConstantVelocity` 這個測試（等速運動下速度要收斂到真值）直接把它逼出來。**沒有這個測試，這個 bug 會一路帶到 Phase 1 的落點計算**，因為落點靠的就是速度外插。
- 教訓：手寫矩陣運算時，「輸出矩陣是否仍然對稱」是最便宜的健檢。
