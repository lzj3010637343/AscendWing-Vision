# ONNX 手术 + ATC 编译指南

> 这份指南教你怎么把一个 **YOLO26 模型**（`.onnx`）「动手术」改成 310B 能跑的样子，再**编译**成板子用的 `.om`。
> 全程复制粘贴即可。每个命令后面都有「预期结果」。

---

## 0. 一句话搞懂我们在干嘛

YOLO26 官方导出的 onnx，最后检测头里用了一个叫 **GatherElements**（二维取数）的算子。**Ascend 310B 不支持这个算子**，直接编译会报错。

「手术」= 把这个官方检测头**剪掉**，换成一个只用 **1D Gather + TopK** 的等效后处理子图（310B 支持），输出还是 `[1, 300, 6]`（300 个检测框，每个 6 个数：类别分、x、y、w、h、置信度）。

```
官方 onnx (445 节点, 含 GatherElements)
        │  yolo26_surgery.py 动手术
        ▼
手术后的 onnx (414 节点, 全是 310B 支持的算子)
        │  ATC 编译
        ▼
板子能跑的 .om (约 20MB)
```

**节点数变化**：445 → 414（剪掉 45 个官方头节点，接上 14 个 TopK 子图节点，净减 31，加上其它清理 = 414）。

---

## 1. 你手上有什么（本包已备齐）

在 `03_onnx_surgery/` 目录里：

| 文件/目录 | 是什么 |
|-----------|--------|
| `yolo26_surgery.py` | **手术脚本**（核心，照着跑就行） |
| `source_onnx/yolo26s_ball_opset11.onnx` | 乒乓球模型**源 onnx**（445 节点，nc=1）— 手术输入 |
| `source_onnx/yolo26s_visdrone_opset11.onnx` | VisDrone 模型**源 onnx**（445 节点，nc=10）— 手术输入 |
| `reference_onnx/yolo26s_visdrone_topk_ref.onnx` | VisDrone 手术后**参考成品**（414 节点）— 用来对答案 |
| `reference_onnx/yolo26s_visdrone_pruned_ref.onnx` | VisDrone 只剪枝、还没接 TopK 的中间态（400 节点）— 供理解 |
| `ascendwing_export_code/aipp_pad32.cfg` | **AIPP 配置**（ATC 编译时用，处理 YUV 输入+pad+归一化） |
| `ascendwing_export_code/aipp_yuv_final.cfg` | 另一份 AIPP 配置（备用） |
| `ascendwing_export_code/*.py` | 推理/预/后处理参考代码（infer/preprocess/postprocess/dvpp） |
| `ascendwing_export_code/labels.txt` | 类别标签 |
| `inspect_onnx.py` | 检查 onnx 结构的小工具（可选） |
| `yolo26cpp-full-reference.md` | 完整原理参考（想深入再看） |

---

## 2. 先跑通手术（用乒乓球模型演示）

### 2.1 进 WSL，进目录
```bash
cd /mnt/d/AscendWing-Package/03_onnx_surgery
```

### 2.2 改脚本里的两个路径
用记事本打开 `yolo26_surgery.py`，改最上面三行：
```python
SRC = '/mnt/d/AscendWing-Package/03_onnx_surgery/source_onnx/yolo26s_ball_opset11.onnx'
DST = '/mnt/d/AscendWing-Package/03_onnx_surgery/yolo26s_ball_topk.onnx'

NA = 8400   # anchors（别改）
NC = 1      # 乒乓球是 1 类，别改；VisDrone 改成 10
TK = 300    # top-k（别改）
```
> 也可以用 `sed` 一行改（不用开编辑器）：
> ```bash
> sed -i \
>   -e "s|^SRC = .*|SRC = '$(pwd)/source_onnx/yolo26s_ball_opset11.onnx'|" \
>   -e "s|^DST = .*|DST = '$(pwd)/yolo26s_ball_topk.onnx'|" \
>   yolo26_surgery.py
> ```

### 2.3 跑手术
```bash
python3 yolo26_surgery.py
```
**预期输出**（关键看最后几行）：
```
连接点: bbox=/model.23/Split_output_0, cls=/model.23/Split_output_1
剪枝: 445 -> 414 节点（剪掉 31 个官方头节点）
initializer: ... -> ...
保存: /mnt/d/.../yolo26s_ball_topk.onnx
最终节点: 414
算子分布: {...}
310B不支持算子残留: GatherElements=0 Tile=0 Mod=0
输出: [('dets', [1, 300, 6])]
onnx.checker: PASS ✅
```
**对答案**：
- `GatherElements=0 Tile=0 Mod=0` —— 三个 310B 不支持的算子全清零 ✅
- `输出: [('dets', [1, 300, 6])]` —— 输出形状对 ✅
- `onnx.checker: PASS ✅` —— 模型合法 ✅

三个都绿，手术成功。生成了 `yolo26s_ball_topk.onnx`（约 38MB）。

### 2.4 （可选）和参考成品对一下结构
```bash
python3 inspect_onnx.py    # 默认看 /tmp 里的参考，需改路径
```
或自己写两行对比：
```bash
python3 - <<'PY'
import onnx
for tag,p in [("我的成品","yolo26s_ball_topk.onnx"),
              ("VisDrone参考","reference_onnx/yolo26s_visdrone_topk_ref.onnx")]:
    m=onnx.load(p); g=m.graph
    print(tag,"nodes=",len(g.node),
          "in=",[(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input],
          "out=",[(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
PY
```
你的成品应当：414 节点、输入 `images[1,3,640,640]`、输出 `dets[1,300,6]`。（类别数不同，内部 cls 维度不同，但结构一致。）

---

## 3. 换成 VisDrone 模型再做一次（10 类）

把脚本三行改成：
```python
SRC = '/mnt/d/AscendWing-Package/03_onnx_surgery/source_onnx/yolo26s_visdrone_opset11.onnx'
DST = '/mnt/d/AscendWing-Package/03_onnx_surgery/yolo26s_visdrone_topk.onnx'
NC = 10      # VisDrone 10 类
```
再 `python3 yolo26_surgery.py`，同样要看到 `GatherElements=0` 和 `PASS ✅`。

> 这个产出的 `yolo26s_visdrone_topk.onnx` 应当和 `reference_onnx/yolo26s_visdrone_topk_ref.onnx` **几乎一样**（414 节点）。可以对一下节点数。

---

## 4. 把 onnx 编译成 .om（ATC）

> 有两个地方能编：**WSL**（需装 CANN，见 `04_environment/WSL_SETUP_GUIDE.md` 第 4 节）或**板子**（已装好，最省事）。下面两条路二选一。

### 路线 A：在板子上编（推荐，零安装）

#### A.1 把手术后的 onnx + aipp 配置传到板子
```bash
# 在 WSL 里（板子和 WSL 都能 SSH 到，因为板子是 USB 虚拟网卡，WSL 也能访问）
scp yolo26s_ball_topk.onnx ascendwing_export_code/aipp_pad32.cfg root@192.168.137.5:/root/
# 输密码：<板子密码>
```
> 如果 WSL 里 ssh/scp 没装：`sudo apt install -y openssh-client`
> 如果 `192.168.137.5` 不通，试 `192.168.137.2`。

#### A.2 SSH 进板子，跑 ATC
```bash
ssh root@192.168.137.5       # 密码 <板子密码>

# 进板子后，先 source CANN 环境（atc 不在默认 PATH）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd /root

# 编译（一行长命令，照抄）
atc --model=yolo26s_ball_topk.onnx \
    --framework=5 \
    --output=yolo26s_ball_topk \
    --input_format=NCHW \
    --input_shape='images:1,3,640,640' \
    --soc_version=Ascend310B1 \
    --insert_op_conf=aipp_pad32.cfg
```
**预期**：跑一两分钟，最后看到：
```
ATC start working now, please wait for a moment.
...
ATC run success
```
生成 `/root/yolo26s_ball_topk.om`（约 20MB）。

> ⚠️ `--framework=5` 表示 ONNX。`--soc_version=Ascend310B1` 是板子的 NPU 型号。`--insert_op_conf=aipp_pad32.cfg` 把 YUV 预处理嵌进模型里（板子喂 YUV 图，模型自己 pad+转 RGB+归一化）。

#### A.3 把 om 拉回本机（可选）
```bash
# 在 WSL / 本机
scp root@192.168.137.5:/root/yolo26s_ball_topk.om /mnt/d/AscendWing-Package/02_weights/
```

### 路线 B：在 WSL 里编（需先装 CANN）

```bash
# WSL 里，先 source CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd /mnt/d/AscendWing-Package/03_onnx_surgery

atc --model=yolo26s_ball_topk.onnx \
    --framework=5 \
    --output=yolo26s_ball_topk \
    --input_format=NCHW \
    --input_shape='images:1,3,640,640' \
    --soc_version=Ascend310B1 \
    --insert_op_conf=ascendwing_export_code/aipp_pad32.cfg
```
同样要看到 `ATC run success`，生成 `yolo26s_ball_topk.om`。

---

## 5. AIPP 配置是干嘛的（`aipp_pad32.cfg`）

不用改，知道它做了啥就行：`aipp_pad32.cfg` 把图像预处理（YUV 输入、色彩转换、缩放 padding、归一化）嵌进模型，由 NPU 硬件执行，板子直接喂摄像头/DVPP 的 YUV 帧。具体参数见 cfg 文件。

---

## 6. 手术原理（想理解再看，不想看跳过也行）

手术脚本 `yolo26_surgery.py` 做了 9 件事：

1. **找连接点**：定位官方检测头里的 `/model.23/Split` 节点，它输出 bbox `[1,8400,4]` 和 cls `[1,8400,NC]`。
2. **反向 BFS 剪枝**：从 Split 的输出往回追溯，保留 Split 和它所有依赖的前置节点；**Split 之后的官方头全删**（GatherElements、Tile、Mod、TopK、Cast、Unsqueeze、Flatten 等约 45 个节点）。
3. 清空原图的 output。
4. 加几个常量（K=300、各种 shape 张量）。
5. **接 14 节点的 1D-gather TopK 子图**，流程：
   ```
   cls[1,8400,NC] --ReduceMax(axis=2)--> scores[8400]
   scores --TopK(K=300)--> vals[300], idx[300]
   bbox[1,8400,4] --Reshape--> [8400,4]
   cls[1,8400,NC] --Reshape--> [8400,NC] --ArgMax(axis=1)--> cls_ids[8400]
   Gather(bbox_2d, idx, axis=0) --> bbox_topk[300,4]   ← 关键：1D Gather, 310B 支持
   Gather(cls_ids, idx, axis=0) --> cls_topk[300]
   Concat(cls_topk_f, bbox_topk, vals) --> [300,6]
   Reshape --> dets[1,300,6]
   ```
6. 设输出 `dets[1,300,6]`。
7. 清掉没人用的遗留常量。
8. 保存。
9. `onnx.checker` 验证。

**关键点**：所有 Gather 都是 **axis=0 的一维取数**，替代官方的二维 GatherElements。这是 310B 能编的根本原因。

> nc=1（乒乓球）时，ArgMax 对 `[8400,1]` 输出全是 0（只有 1 类，恒为类 0），不用改结构，天然兼容。

---

## 7. 常见坑

| 现象 | 解决 |
|------|------|
| `未找到 /model.23/Split` | 源 onnx 不是标准 yolo26 导出（可能已剪过头）。换 `source_onnx/` 里的源文件 |
| 手术后 `GatherElements` 不是 0 | 源 onnx 头结构变了，脚本要调（看 `yolo26cpp-full-reference.md`） |
| ATC 报算子不支持 | 手术没做干净，回到第 2.3 看 `GatherElements=0 Tile=0 Mod=0` 是否成立 |
| ATC 报 `input_shape` 不匹配 | 必须是 `images:1,3,640,640`，且 onnx 输入名是 `images` |
| ATC 报 `aipp` 错 | `--insert_op_conf=` 后的路径要对；cfg 里 `input_format` 要和实际喂的图一致 |
| 板子上 `atc: command not found` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| om 在板子上加载报错 | 两个 om 字节数几乎一样，**别混用**（见 `02_weights/WEIGHTS_README.md`） |

---

## 8. 产物对照表

| 模型 | 源 onnx | NC | 手术后 onnx | 编译后 om | om 大小 |
|------|---------|----|-----------|----------|--------|
| 乒乓球 | `source_onnx/yolo26s_ball_opset11.onnx` | 1 | `yolo26s_ball_topk.onnx` | `yolo26s_ball_topk.om` | 20,835,428 字节 |
| VisDrone | `source_onnx/yolo26s_visdrone_opset11.onnx` | 10 | `yolo26s_visdrone_topk.onnx` | `yolo26s_yuv_pad32_topk.om` | 20,790,291 字节 |

> 包里 `02_weights/` 已经放好了这两个**编译好的 om**，不想自己编可以直接用。
