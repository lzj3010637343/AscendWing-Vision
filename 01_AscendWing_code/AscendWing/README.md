# AscendWing

YOLO26 real-time object detection + SOT tracking on Ascend 310B.

## Architectures

### Multi-Process (display & tracking separated)

```
Process 1 (C++, ./run.sh unified):
  Camera -> NPU inference -> AShmSyncFrame -> /sync_frame_shm

Process 2 (Python, ./run.sh sot):
  Read /sync_frame_shm -> SOT tracking -> Draw boxes -> MJPEG :8081
  -> Write SOT result to /sot_control_shm
```

### IPC (original - single process, backward compatible)

```
Camera -> ascendwing_tracker (NPU + MJPEG)
           ├── /detection_shm -> detection_reader -> JSON
           ├── /image_shm -> image_reader -> JPEG
           │      └── mjpeg_server_standalone.py -> HTTP :8080
           └── Built-in MJPEG :8080
```

## Build

```bash
make -j$(nproc) install
```

Depends on: mxVision 26.0.0 + CANN 9.0.0 + Ascend 310B driver.

## Run - Unified + SOT（实际使用，两进程）

详见 [STARTUP.md](STARTUP.md)。两个终端按顺序：

```bash
Term 1:  ./run.sh sot                                  # SOT 跟踪 + MJPEG :8081
Term 2:  ./run.sh unified /root/yolo26s_ball_topk.om   # C++ YOLO -> sync shm

# Browser: http://192.168.137.5:8081/
./stop.sh
```

## Run - Multi-Process

```bash
Term 1:  ./run.sh tracker              # C++ NPU inference + sync shm
Term 2:  ./run.sh sot                  # Python SOT + MJPEG :8081

# Browser: http://192.168.137.5:8081/

./stop.sh   # kill all + clean shm
```

## Run - IPC (Legacy)

```bash
Term 1:  ./run.sh tracker
Term 2:  ./run.sh det_reader --benchmark
Term 3:  ./run.sh mjpeg

# Browser: http://192.168.137.5:8080/

./stop.sh
```

## SOT Algorithm

改进的 SOT 单目标跟踪，实现见 `src/sot/single_object_tracker.py`。

## Structure

```
AscendWing/
├── Makefile
├── run.sh / stop.sh
├── src/
│   ├── tracker/             C++ camera + DVPP + NPU
│   │   ├── main.cpp         IPC tracker (writes sync shm + legacy shm)
│   │   ├── main_unified.cpp Unified tracker (writes sync shm)
│   │   ├── mx_wrapper.cpp/h mxVision inference wrapper
│   │   ├── mjpeg_server.h   Threaded MJPEG HTTP server
│   │   └── eis.h            Electronic image stabilization
│   ├── sot/                 Python SOT tracker
│   │   ├── sot_live.py          Multi-threaded reader + SOT + MJPEG
│   │   └── single_object_tracker.py  Improved SOT algorithm
│   └── ipc/                 POSIX shm IPC
│       ├── ascend_shm.c/h       Legacy separate det+img queues
│       └── sync_frame_shm.c/h   Combined AShmSyncFrame + SOTControlData
└── install/bin/             Build output
```

## IPC Performance

| Metric | Value |
|--------|-------|
| POSIX shm memcpy+sem_post | <10 μs |
| AShmSyncFrame size | 525,524 bytes |
| SOTControlData size | 40 bytes |
| Tracker FPS (310B1) | ~70 fps |
| NPU inference (yolo26s) | ~10 ms |
