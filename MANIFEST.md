# MANIFEST - AscendWing 视觉包文件清单

打包时间：2026-07-23（原包）｜ 2026-07-28 剥离视觉开源 ｜ 总大小：**~186 MB** ｜ 文件数：56

> md5 校验见 `02_weights/WEIGHTS_README.md`。目录树概览见 `README.md`。

---

## 01_AscendWing_code/（~420 KB）- 板上视觉源码

从板子 `/root/AscendWing` tar 拉取，**已剔除 build/ install/ __pycache__/ 所有 .om，以及控制代码与历史版本目录**。

| 路径 | 说明 |
|------|------|
| `README.md` | 项目架构总览（2 进程视觉管线、性能、结构） |
| `STARTUP.md` | 视觉启动序列（2 终端命令、shm） |
| `Makefile` | C++ 编译（mxVision 26.0.0 + CANN 9.0.0） |
| `run.sh` / `stop.sh` | 启动/停止脚本（sot/unified/tracker/det_reader/img_reader/mjpeg/vision） |
| `aipp_rgb_norm.cfg` / `aipp_yuv_pad32.cfg` | AIPP 配置（板子版） |
| `scripts/loopback_test.py` | 视频回环测试 |
| `src/tracker/` | **C++ NPU 推理核**：main.cpp, main_unified.cpp, mx_wrapper.cpp/h, mx_model.cpp/h, sot_tracker.h, mjpeg_server.h, eis.h, vision_node.cpp |
| `src/sot/` | **Python SOT 跟踪**：sot_live.py, single_object_tracker.py, TUNING.md, __init__.py |
| `src/ipc/` | **C 共享内存 IPC**：ascend_shm.c/h, sync_frame_shm.c/h, image_reader.c, detection_reader.c |
| `models/` | 空（om 单独在 02_weights/） |

---

## 02_weights/ - YOLO26 OM 权重（从 Release 下，见 WEIGHTS_README）

> `.om` 文件较大，不随仓库分发。从 [Release](../releases) 下 `weights.zip`，解压后 `weights/*.om` 放到此目录。

| 文件 | 大小 | md5 | 说明 |
|------|------|-----|------|
| `yolo26s_yuv_pad32_topk.om` | 20,790,291 | `86336a8c973fbe75303918f5cdb10efe` | **VisDrone 10 类**，通用检测 |
| `yolo26s_ball_topk.om` | 20,835,428 | `fb681a2253bed78fdb6f8ec060d152b5` | **乒乓球 nc=1** |
| `WEIGHTS_README.md` | 2.3K | - | 下载说明 + 区别 + 验证 + 部署 |

两者同手术 + 同 AIPP，字节数几乎一样，**按文件名区分**。

---

## 03_onnx_surgery/ - ONNX 手术 + ATC 全套

> `.onnx` 文件较大，不随仓库分发。从 [Release](../releases) 下 `weights.zip`，解压后 `source_onnx/*.onnx` 和 `reference_onnx/*.onnx` 放到对应目录。

| 路径 | 大小 | 说明 |
|------|------|------|
| `SURGERY_GUIDE.md` | 11K | **手术+ATC 步骤**（必读） |
| `yolo26_surgery.py` | 4.8K | **手术脚本**：445->414 节点，接 1D-gather TopK 子图 |
| `inspect_onnx.py` | 1.8K | onnx 结构检查小工具 |
| `ascendwing_export_code/aipp_pad32.cfg` | 0.7K | **AIPP 配置**（DVPP+硬件预处理，参数见 cfg） |
| `ascendwing_export_code/aipp_yuv_final.cfg` | 0.6K | 备用 AIPP |
| `ascendwing_export_code/infer.py` | 3.8K | 推理参考 |
| `ascendwing_export_code/preprocess.py` | 1.2K | 预处理参考 |
| `ascendwing_export_code/postprocess.py` | 1.7K | 后处理参考 |
| `ascendwing_export_code/dvpp_preprocess.py` | 5.8K | DVPP 预处理参考 |
| `ascendwing_export_code/labels.txt` | 0.1K | VisDrone 10 类标签 |
| `source_onnx/yolo26s_ball_opset11.onnx` | 38M | **乒乓球源 onnx**（445 节点，nc=1，手术输入，从 Release 下） |
| `source_onnx/yolo26s_visdrone_opset11.onnx` | 38M | **VisDrone 源 onnx**（445 节点，nc=10，手术输入，从 Release 下） |
| `reference_onnx/yolo26s_visdrone_topk_ref.onnx` | 38M | VisDrone 手术后**参考成品**（414 节点，对答案，从 Release 下） |
| `reference_onnx/yolo26s_visdrone_pruned_ref.onnx` | 38M | VisDrone 只剪枝中间态（400 节点，从 Release 下） |

> onnx 结构已核验：源 = 445 节点 `output0[1,300,6]`；参考成品 = 414 节点 `dets[1,300,6]`。

---

## 04_environment/（12 KB）- 环境信息

| 文件 | 大小 | 说明 |
|------|------|------|
| `WSL_SETUP_GUIDE.md` | 7.5K | ** WSL+CANN+ATC 环境搭建**（必读） |
| `board_env.txt` | 3.3K | 板子环境快照（OS/kernel/python/CANN/mxIndex/npu-smi/磁盘） |

板子：openEuler 22.03 / kernel 6.6.0 ascend aarch64 / Python 3.9.9 / CANN 9.0.0 / mxIndex 26.0.0 / npu-smi 26.0.rc1 / 310B1。
打包机 WSL：Ubuntu 24.04 x86_64 / onnx 1.22.0 / CANN 9.1.0-beta.1。

---

---

## 顶层

| 文件 | 说明 |
|------|------|
| `README.md` | **总入口**（总流程，先看这个） |
| `MANIFEST.md` | 本文件 |
