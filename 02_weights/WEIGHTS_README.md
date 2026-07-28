# YOLO26 权重说明

两个 **310B 能直接跑的 `.om` 模型**，YOLO26s + TopK 手术后编译（手术见 `03_onnx_surgery/SURGERY_GUIDE.md`）。

> ⚠️ `.om` 和 `.onnx` 权重文件较大（共约 186 MB），**不随仓库分发**。请从 [GitHub Release](../../releases) 下载 `weights.zip`，解压后按目录结构放回：
> - `weights/*.om` → 放到 `02_weights/`
> - `source_onnx/*.onnx` → 放到 `03_onnx_surgery/source_onnx/`
> - `reference_onnx/*.onnx` → 放到 `03_onnx_surgery/reference_onnx/`

## 两个 om

| 文件名 | 用途 | 类别数 NC | 输入 | 输出 | 大小（字节） | md5 |
|--------|------|----------|------|------|------------|-----|
| `yolo26s_yuv_pad32_topk.om` | **VisDrone 10 类**（人/车/…）通用检测 | 10 | `images[1,3,640,640]`（DVPP+AIPP） | `dets[1,300,6]` | 20,790,291 | `86336a8c973fbe75303918f5cdb10efe` |
| `yolo26s_ball_topk.om` | **乒乓球 nc=1** 检测 | 1 | 同上 | `dets[1,300,6]` | 20,835,428 | `fb681a2253bed78fdb6f8ec060d152b5` |

> 输出 `dets[1,300,6]` = 最多 300 个框，每个 `[class_id, x, y, w, h, score]`（顺序以实际推理代码为准，见 `01_AscendWing_code/.../postprocess.py`）。

## ⚠️ 两个文件几乎一样大，别混用

字节数只差 4 万（20.79M vs 20.84M），**靠文件名区分**。混用不会报错，但：
- 拿乒乓球 om 检测人/车 → 一个都检不到
- 拿 VisDrone om 检测球 → 类别错或 不报

## 校验文件没损坏

```bash
md5sum yolo26s_yuv_pad32_topk.om   # 应为 86336a8c973fbe75303918f5cdb10efe
md5sum yolo26s_ball_topk.om        # 应为 fb681a2253bed78fdb6f8ec060d152b5
```
md5 对不上 = 文件传坏了或被改过，重新下/传。

## 部署到板子

```bash
scp yolo26s_ball_topk.om root@192.168.137.5:/root/
scp yolo26s_yuv_pad32_topk.om root@192.168.137.5:/root/
# 密码 <板子密码>
```

启动时指定模型（`run.sh unified <model>`）：
```bash
cd /root/AscendWing
./run.sh unified /root/yolo26s_ball_topk.om      # 跟球
./run.sh unified /root/yolo26s_yuv_pad32_topk.om  # 通用检测
```

## 想自己重新编？

源 onnx + 手术脚本 + ATC 命令全在 `03_onnx_surgery/`，照着 `SURGERY_GUIDE.md` 走即可重新生成这两个 om。
