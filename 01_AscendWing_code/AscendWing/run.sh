#!/bin/bash
# AscendWing — YOLO26 + POSIX shm IPC on Ascend 310B
#
# Usage:
#   ./run.sh tracker [model_path]
#   ./run.sh det_reader [--benchmark] [--poll]
#   ./run.sh img_reader [--benchmark] [--poll]
#   ./run.sh mjpeg
#   ./run.sh sot [--port 8081]        # SOT + MJPEG :8081
#   ./run.sh multi                    # Show multi-process startup
#   ./run.sh all                      # Show IPC startup

set -e
cd "$(dirname "$0")"

export MX_SDK_HOME=/root/mxVision-26.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64:${LD_LIBRARY_PATH}"

CMD="$1"
shift 2>/dev/null || true

case "$CMD" in
    tracker|ipc)
        make -j$(nproc) install
        v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=70 -c gain=40 \
                 -c sharpness=0 -c contrast=5 -c gamma=90 2>/dev/null &
        exec ./install/bin/ascendwing_tracker "$@"
        ;;
    unified|uni)
        make -j$(nproc) install
        v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=70 -c gain=40 \
                 -c sharpness=0 -c contrast=5 -c gamma=90 2>/dev/null &
        exec ./install/bin/ascendwing_unified "$@"
        ;;
    det|det_reader|detection_reader)
        make -j$(nproc) install
        exec ./install/bin/detection_reader "$@"
        ;;
    img|img_reader|image_reader)
        make -j$(nproc) install
        exec ./install/bin/image_reader "$@"
        ;;
    mjpeg)
        exec python3 /root/catkin_ws/src/yolov5cpp/scripts/mjpeg_server_standalone.py \
            --reader_bin "$PWD/install/bin/image_reader" "$@"
        ;;
    sot|sot_live)
        exec python3 "$PWD/src/sot/sot_live.py" "$@"
        ;;
    vision|vision_node)
        make -j$(nproc) install
        v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=70 -c gain=40 \
                 -c sharpness=0 -c contrast=5 -c gamma=90 2>/dev/null &
        # LD_PRELOAD: resolve GStreamer symbol conflict (same as yolov5cpp)
        SYS_GST=$(find /usr/lib /lib -name "libgstreamer-1.0.so.0" 2>/dev/null | grep "aarch64-linux-gnu" | head -n 1)
        exec env LD_PRELOAD="${SYS_GST}" ./install/bin/vision_node "$@"
        ;;
    multi|multiprocess)
        echo "=== AscendWing Vision Pipelines ==="
        echo ""
        echo "-- Unified (general objects) --"
        echo "Term 1:  ./run.sh unified        # C++ YOLO → sync shm"
        echo "Term 2:  ./run.sh sot             # Python SOT + MJPEG :8081"
        echo ""
        echo "Browser: http://192.168.137.5:8081/"
        echo ""
        echo "SHM channels:"
        echo "  /dev/shm/sync_frame_shm   (unified → display, det+JPEG)"
        echo "  /dev/shm/sot_control_shm  (unified → downstream, SOT result)"
        exit 0
        ;;
    all)
        echo "=== AscendWing IPC Pipeline ==="
        echo ""
        echo "Term 1:  ./run.sh tracker"
        echo "Term 2:  ./run.sh det_reader --benchmark"
        echo "Term 3:  ./run.sh mjpeg"
        echo ""
        echo "Browser: http://192.168.137.5:8080/"
        exit 0
        ;;
    *)
        echo "Usage: $0 {tracker|det_reader|img_reader|mjpeg|sot|vision|multi|all} [args...]"
        echo ""
        echo "IPC modes (original):"
        echo "  tracker      Camera -> NPU -> POSIX shm + MJPEG :8080"
        echo "  det_reader   POSIX shm -> JSON stdout"
        echo "  img_reader   POSIX shm -> JPEG stdout"
        echo "  mjpeg        HTTP MJPEG stream on :8080"
        echo "  all          Show IPC startup sequence"
        echo ""
        echo "Multi-process modes:"
        echo "  sot          SOT tracking + MJPEG :8081"
        echo "  vision       YOLO detection + tracking → /sot_control_shm"
        echo "  multi        Show pipeline startup sequences"
        exit 1
        ;;
esac
