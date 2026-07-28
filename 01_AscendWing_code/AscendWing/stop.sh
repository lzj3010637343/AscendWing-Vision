#!/bin/bash
# AscendWing 停止脚本 — 清理所有进程 + POSIX shm 残留

echo "[stop] Killing C++ tracker..."
pkill -f ascendwing_tracker 2>/dev/null || true

echo "[stop] Killing readers..."
pkill -f detection_reader 2>/dev/null || true
pkill -f image_reader 2>/dev/null || true

echo "[stop] Killing Python processes..."
pkill -f mjpeg_server_standalone 2>/dev/null || true
pkill -f sot_live 2>/dev/null || true

echo "[stop] Cleaning POSIX shm..."
rm -f /dev/shm/detection_shm /dev/shm/detection_sem \
      /dev/shm/image_shm    /dev/shm/image_sem \
      /dev/shm/sync_frame_shm /dev/shm/sync_frame_sem \
      /dev/shm/sot_control_shm 2>/dev/null || true

echo "[stop] Restoring camera auto-exposure..."
v4l2-ctl -d /dev/video0 -c exposure_auto=3 2>/dev/null || true

echo "[stop] Done."
