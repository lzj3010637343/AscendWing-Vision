# AscendWing 视觉启动序列

> 双进程管线: sot(SOT 跟踪 + MJPEG 推流) + unified(C++ YOLO 检测)
> 工作目录: /root/AscendWing

## 两个命令（各开一个终端，按顺序）

```bash
# Term 1: SOT 跟踪 + MJPEG 推流
./run.sh sot

# Term 2: C++ YOLO 检测 -> /sot_control_shm（指定球模型 + TopK 后处理）
./run.sh unified /root/yolo26s_ball_topk.om
```

> 顺序说明: sot 和 unified 都读写同一份 shm，先起哪个都行，但建议 sot 先起（等 unified 写 shm 后才有数据）。

## 各进程作用 + 文件路径

| 进程 | 命令 | 作用 | 源文件 | 输出 |
|------|------|------|--------|-----------|
| SOT | `./run.sh sot` | 读 sync shm，SOT 跟踪，画框，MJPEG 推流 | `/root/AscendWing/src/sot/sot_live.py` | MJPEG :8081, 写 /sot_control_shm |
| Unified | `./run.sh unified <model>` | C++ NPU YOLO 检测+跟踪，写 sync shm | `/root/AscendWing/install/bin/ascendwing_unified`（run.sh 自动 make） | /dev/shm/sync_frame_shm, /dev/shm/sot_control_shm |

## 模型

- `/root/yolo26s_ball_topk.om` - 球检测 + TopK 后处理（Term 2 unified 用，20MB）
- `/root/yolo26s_yuv_pad32_topk.om` - VisDrone 10 类通用检测（可选）

## 共享内存

| 路径 | 写者 | 读者 | 大小 |
|------|------|------|------|
| /dev/shm/sync_frame_shm | unified | sot | ~525KB |
| /dev/shm/sot_control_shm | unified/sot | 下游消费者（可选） | 40B |

> /sot_control_shm 输出 SOT 跟踪结果（目标中心、框、置信度、状态），供下游应用读取。本视觉包自身不消费它，仅作为输出接口保留。

## 停止

```bash
cd /root/AscendWing && ./stop.sh   # 杀所有进程 + 清 shm
```

## 浏览器

http://192.168.137.5:8081/  （SOT MJPEG 推流）
