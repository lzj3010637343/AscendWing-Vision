# AscendWing 视觉包：YOLO26 检测 + SOT 跟踪 + ONNX 手术 + 环境

> 这个包是从 **Ascend 310B** 板子（IP `192.168.137.5`）上导出的**实时视觉**工作集。
> 包含：AscendWing 视觉代码（YOLO26 检测 + SOT 跟踪）、两个 YOLO26 OM 权重、ONNX 手术全套素材、板子/WSL 环境信息。
> 目标：让一个从没碰过这套东西的人，照着文档也能把环境搭起来、把模型跑起来。

---

## 关于这个开源版本

2026 年第 21 届研电赛「昇羽」队 AscendWing 代码的**视觉 + SOT 部分**开源。
主要工作是在 Ascend 310B（20T 版本）上快速部署 YOLO26，推理管线约 **70 FPS**。
代码由 GLM 辅助编写，**仅作为教学用途**。

---

## 效果预览

![AscendWing 视觉跟踪演示](assets/preview.gif)

*310B 板载 YOLO26 检测 + SOT 跟踪实时推流（浏览器 :8081，~60 FPS）*

---

## 这个项目是干嘛的？（30 秒版）

AscendWing = 在 **Ascend 310B** 板子上跑 **YOLO26 实时检测 + SOT 跟踪**，把摄像头画面里的目标检测出来、锁住跟踪、浏览器看推流。

板子上是 2 个进程配合：
```
摄像头 -> [C++ YOLO 检测 NPU] -> 共享内存 -> [Python SOT 跟踪 + 画框 + MJPEG推流]
```
浏览器看推流：`http://192.168.137.5:8081/`

详细架构看 `01_AscendWing_code/AscendWing/README.md` 和 `STARTUP.md`。

---

## 包里有什么

```
AscendVision-Package/
├── README.md                       ← 你正在看
├── MANIFEST.md                     ← 全部文件清单
│
├── 01_AscendWing_code/AscendWing/  ← 板子上的源码（去掉 build 和大 om）
│   ├── README.md / STARTUP.md      ← 架构 + 启动步骤（必读）
│   ├── Makefile / run.sh / stop.sh
│   ├── src/
│   │   ├── tracker/   C++ 摄像头+DVPP+NPU 推理
│   │   ├── sot/       Python SOT 跟踪
│   │   └── ipc/       C 共享内存 IPC
│   └── scripts/
│
├── 02_weights/                     ← YOLO26 OM 权重（从 Release 下，见 WEIGHTS_README）
│   └── WEIGHTS_README.md           ← 下载说明 + 区别 + 验证 + 部署
│
├── 03_onnx_surgery/                ← ONNX 手术 + ATC 编译全套
│   ├── SURGERY_GUIDE.md            ← 手术+编译步骤（必读）
│   ├── yolo26_surgery.py           ← 手术脚本
│   ├── source_onnx/                ← 源 onnx（手术输入，从 Release 下）
│   ├── reference_onnx/             ← 手术后参考成品（对答案，从 Release 下）
│   ├── ascendwing_export_code/     ← AIPP 配置 + 推理参考代码
│   └── inspect_onnx.py             ← onnx 检查小工具
│
└── 04_environment/                 ← 环境信息
    ├── WSL_SETUP_GUIDE.md          ←  WSL+CANN+ATC 环境搭建（必读）
    └── board_env.txt               ← 板子环境快照
```

---

## 你想干什么？（对号入座）

### 🅰️ 我只想在板子上跑 AscendWing 视觉（部署/演示）
不用装 WSL。直接：
1. 把 `01_AscendWing_code/AscendWing/` 整个传到板子 `/root/`
2. 把 `02_weights/yolo26s_ball_topk.om` 传到板子 `/root/`
3. 板子上 `cd /root/AscendWing`，照 `STARTUP.md` 开 2 个终端跑
-> 看 `01_AscendWing_code/AscendWing/STARTUP.md`

### 🅱️ 我想重新编译/改造 YOLO26 模型（手术 + ATC）
1. 先装 WSL：看 `04_environment/WSL_SETUP_GUIDE.md`（装 onnx 即可，5 分钟）
2. 跑手术 + 编译：看 `03_onnx_surgery/SURGERY_GUIDE.md`
3. 产物 om 放进 `02_weights/`，再按 🅰️ 部署

### 🅲 我只想了解原理（看文档）
读 `03_onnx_surgery/SURGERY_GUIDE.md`（ONNX 手术原理）和 `01_AscendWing_code/AscendWing/README.md`（架构总览）。

---

## 最快上手（3 步，约 10 分钟，前提是板子在手边）

```bash
# ① 装好 WSL + onnx（第一次才需要，见 04_environment/WSL_SETUP_GUIDE.md）
#    本机已装好可跳过

# ② 在 WSL 里跑一次手术，确认链路通（先从 Release 下 source_onnx 放到 03_onnx_surgery/source_onnx/）
cd /mnt/d/AscendVision-Package/03_onnx_surgery
sed -i \
  -e "s|^SRC = .*|SRC = '$(pwd)/source_onnx/yolo26s_ball_opset11.onnx'|" \
  -e "s|^DST = .*|DST = '$(pwd)/yolo26s_ball_topk.onnx'|" \
  yolo26_surgery.py
python3 yolo26_surgery.py
# 看到 "onnx.checker: PASS ✅" 就成功

# ③ 用现成 om 部署到板子（先从 Release 下 om 放到 02_weights/）
scp /mnt/d/AscendVision-Package/02_weights/yolo26s_ball_topk.om root@192.168.137.5:/root/
scp -r /mnt/d/AscendVision-Package/01_AscendWing_code/AscendWing root@192.168.137.5:/root/
ssh root@192.168.137.5   # 密码 <板子密码>
# 板子上：
cd /root/AscendWing && ./run.sh sot        # 终端1
./run.sh unified /root/yolo26s_ball_topk.om # 终端2
# 浏览器开 http://192.168.137.5:8081/
```

---

## 板子怎么连

| 项 | 值 |
|----|----|
| IP | `192.168.137.5`（USB RNDIS 虚拟网卡；偶尔变 `.2`，两个都试） |
| SSH | `ssh root@192.168.137.5`，端口 22 |
| 用户/密码 | `root` / `<板子密码>` |
| 连法 | USB 数据线连板子 TYPE-C/USB 口，板子开机后会虚拟出网卡，Windows 自动装驱动 |
| 推流地址 | `http://192.168.137.5:8081/`（SOT MJPEG） |

板子环境详情：`04_environment/board_env.txt`。

---

## ⚠️ 几个必须知道的坑

1. **两个 om 别混用**：`yolo26s_ball_topk.om`（乒乓球）和 `yolo26s_yuv_pad32_topk.om`（VisDrone）字节数几乎一样，靠**文件名**区分。详见 `02_weights/WEIGHTS_README.md`。
2. **ATC 只能在 Linux 跑**：Windows 不行。用 WSL 或直接用板子（板子已装好 CANN+atc，最省事）。
3. **ONNX 手术必须做**：YOLO26 官方 onnx 含 `GatherElements`（2D 取数），310B 不支持，不手术直接 ATC 会报错。手术把 445 节点改成 414 节点。
4. **本视觉包不依赖 ROS / 外部控制链路**：纯 YOLO 检测 + SOT 跟踪，走 POSIX 共享内存 IPC。别去找 `/opt/ros`。
5. **run.sh 会自动 make**：C++ 部分第一次跑会自动编译（要几分钟），依赖板子上的 mxVision 26.0.0 + CANN 9.0.0。

---

## 版本信息（打包时 2026-07-23）

- 板子：openEuler 22.03 / kernel 6.6.0 ascend aarch64 / Python 3.9.9 / CANN 9.0.0 / mxIndex 26.0.0 / npu-smi 26.0.rc1 / NPU 310B1
- 打包机 WSL：Ubuntu 24.04 x86_64 / onnx 1.22.0 / CANN 9.1.0-beta.1
- 两个 om 的 md5 见 `02_weights/WEIGHTS_README.md`
