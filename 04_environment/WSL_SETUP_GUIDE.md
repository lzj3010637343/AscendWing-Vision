# WSL 环境搭建指南

> 目标：在一台 Windows 电脑上装好 WSL（Windows 里的 Linux），用来跑 **ONNX 手术脚本** 和（可选）**ATC 编译 OM**。
> 你只需要照着一步步复制粘贴，不需要懂原理。每一步都有「预期结果」，照着对一下就行。

---

## 0. 先搞清楚：为什么需要 WSL？

Ascend 310B 板子上的模型文件是 `.om` 格式。把 `.onnx` 变成 `.om` 要用一个叫 **ATC** 的编译器，它**只能在 Linux 上跑**（Windows 跑不了）。

所以我们需要一个 Linux 环境。有两个选择：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A. WSL（本指南）** | 在你 Windows 电脑上直接开 Linux，不用联网到板子，改代码方便 | 要装 CANN 才能编 OM（第 4 节，稍麻烦） |
| **B. 直接用板子** | 板子已装好 CANN + onnx，**零安装**，最省事 | 是部署目标，不适合长期开发 |

**推荐组合**：
- **ONNX 手术**（改模型结构）-> 在 **WSL** 里做（只要装 onnx，秒装）
- **ATC 编译 OM**（onnx->om）-> 图省事就**在板子上做**（第 5 节）；想全程 WSL 就装 CANN（第 4 节）

> 本机（打包者的电脑）已经装好 WSL Ubuntu-24.04 + CANN 9.1.0-beta.1 + onnx 1.22.0，可直接用。下面是给**全新电脑**从头装的步骤。

---

## 1. 装 WSL2 + Ubuntu 24.04

### 1.1 以管理员身份打开 PowerShell
按 `Win` 键 -> 输入 `powershell` -> 右键「Windows PowerShell」->「以管理员身份运行」。

### 1.2 一行命令装好 WSL + Ubuntu
在 PowerShell 里粘贴：
```powershell
wsl --install -d Ubuntu-24.04
```
- 这会自动启用 WSL2 子系统 + 下载 Ubuntu 24.04。
- 装完会提示**重启电脑**，重启。
- 重启后会自动弹出 Ubuntu 窗口，让你设一个**用户名和密码**（随便设，记住密码就行，这是 Linux 里的，跟 Windows 无关）。

**预期结果**：能在开始菜单搜到「Ubuntu」，打开后看到一个绿色/白色提示符的 Linux 终端。

### 1.3 以后怎么进 WSL
- 方法1：开始菜单搜「Ubuntu」打开
- 方法2：在 PowerShell 或 CMD 里输 `wsl` 回车
- 方法3：Windows Terminal 里下拉选 Ubuntu

---

## 2. （在 WSL 里）装 Python 依赖 -- 跑手术脚本用

> Ubuntu 24.04 自带 Python 3.12。手术脚本只需要 `onnx` 和 `numpy` 两个包。

打开 WSL 终端，依次粘贴：

```bash
# 更新软件源（第一次用建议跑一下，可能要一两分钟）
sudo apt update && sudo apt install -y python3-pip python3-venv

# 装 onnx 和 numpy（手术脚本就靠这俩）
pip3 install onnx numpy --break-system-packages
```
> 说明：Ubuntu 24.04 默认不让 pip 装系统级包，加 `--break-system-packages` 就行（或用 venv，见下面「进阶」）。

**验证（粘贴这一段，看到版本号就成功）**：
```bash
python3 -c "import onnx, numpy; print('onnx', onnx.__version__, 'numpy', numpy.__version__)"
```
**预期输出**类似：
```
onnx 1.22.0 numpy 1.26.4
```

到这一步，**ONNX 手术已经能跑了**（跳到 `03_onnx_surgery/SURGERY_GUIDE.md`）。
下面的第 4 节是给「想在 WSL 里编译 OM」的人，**不需要的话可以跳过**。

---

## 3. 把本包的文件放进 WSL

WSL 里访问 Windows 的 D 盘路径是 `/mnt/d/`。本包在 `D:\AscendWing-Package\`，对应 WSL 里 `/mnt/d/AscendWing-Package/`。

```bash
# 进到手术目录
cd /mnt/d/AscendWing-Package/03_onnx_surgery
ls                    # 应该看到 yolo26_surgery.py、source_onnx/、aipp cfg 等
```

---

## 4. （可选）装 CANN 工具链 -- 想在 WSL 里编 OM 才需要

> 只跑手术脚本不用装这节。只有想把 `.onnx` 编译成 `.om` 且不想用板子时才装。
> 这一步比较重（下载约 2GB + 注册账号），**新手建议直接用板子编（第 5 节）**，跳过本节。

### 4.1 下载 CANN toolkit
1. 浏览器打开华为昇腾社区：https://www.hiascend.com/
2. 注册/登录账号 -> 「资源中心」->「软件下载」-> 找 **CANN 商用版**
3. 选 **CANN Toolkit**，系统选 **Linux x86_64**，版本选 **8.0.RC3 / 9.0.0**（跟板子的 CANN 9.0.0 对齐；9.1.0 beta 也行）
4. 下载得到一个 `.run` 文件，比如 `Ascend-cann-toolkit_9.0.0_linux-x86_64.run`
   > 下载页面会让你勾选「同意协议」，还会要求填用途。照实填即可。

### 4.2 装 CANN
把下载的 `.run` 文件放进 WSL（比如复制到 `~/`），然后：
```bash
# 进到 .run 文件所在目录
cd ~

# 加可执行权限
chmod +x Ascend-cann-toolkit_9.0.0_linux-x86_64.run

# 安装（要几分钟，会解压一大堆东西到 /usr/local/Ascend/）
sudo ./Ascend-cann-toolkit_9.0.0_linux-x86_64.run --install

# 装完，source 一下环境变量（每次新开终端都要 source，建议写进 ~/.bashrc）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
```

### 4.3 验证 ATC
```bash
atc --help 2>&1 | head -5
```
**预期**：打印出 ATC 的帮助说明（一堆参数），不报「command not found」就成功。

> ⚠️ ATC 在 x86 WSL 上是**交叉编译**：你的电脑没有 NPU，但 ATC 照样能把 onnx 编成 310B 能跑的 om（它只是个编译器）。所以不用在 Windows 装 NPU 驱动。

---

## 5. （备选）直接用板子编译 OM -- 最省事

板子（192.168.137.5）已经装好 CANN 9.0.0 + onnx 1.19.1 + atc，**什么都不用装**。
如果你嫌 WSL 装 CANN 麻烦，就 SCP 文件到板子上编。详见 `03_onnx_surgery/SURGERY_GUIDE.md` 第 4 节，有完整命令。

板子连接方式（USB 线连电脑，板子会虚拟出网卡）：
- SSH：`ssh root@192.168.137.5`，密码 `<板子密码>`
- （有时板子 DHCP 会拿到 `.2` 而不是 `.5`，两个都试）

---

## 6. 常见坑

| 现象 | 原因 / 解决 |
|------|------|
| `wsl --install` 报错 | Windows 版本太老，先 `wsl --update`；或控制面板「启用或关闭 Windows 功能」勾选「虚拟机平台」+「适用于 Linux 的 Windows 子系统」 |
| `pip3 install` 报 `externally-managed-environment` | 加 `--break-system-packages`，或用 venv（见下） |
| `atc: command not found` | 没 source 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| ATC 报 `soc_version Ascend310B1 not support` | CANN 版本太老，换 8.0.RC3 或 9.0.0+ |
| WSL 访问 D 盘找不到 | 路径是 `/mnt/d/`（小写 d），不是 `D:\` |
| 板子 SSH 连不上 | USB 线没插好 / 板子没开机 / IP 变了，试 `192.168.137.2` |

### 进阶：用虚拟环境 venv（干净，推荐熟手用）
```bash
cd /mnt/d/AscendWing-Package/03_onnx_surgery
python3 -m venv venv
source venv/bin/activate
pip install onnx numpy
# 以后每次进来先 source venv/bin/activate 再跑脚本
```

---

## 环境信息备忘（打包时采集）

**板子（部署目标）**：
- 系统：openEuler 22.03 LTS（aarch64）
- 内核：6.6.0 ascend
- Python 3.9.9，numpy 1.24.4，onnx 1.19.1
- CANN 9.0.0（+ 9.1.0-beta.1），mxIndex/mxVision 26.0.0，npu-smi 26.0.rc1，NPU 310B1
- 不依赖 ROS

**打包机 WSL（开发环境）**：
- Ubuntu 24.04.4 LTS（x86_64）
- onnx 1.22.0，numpy 1.26.4
- CANN cann-9.1.0-beta.1，atc 在 `/usr/local/Ascend/cann-9.1.0-beta.1/bin/atc`

完整环境快照见同目录 `board_env.txt`。
