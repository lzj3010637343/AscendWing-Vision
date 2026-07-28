#pragma once
#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ---- 模型类型枚举 ----
enum ModelSubtype {
    SUBTYPE_TOPK,         // [1,300,6] TopK 输出 (yolo26s_yuv_pad32_topk)
    SUBTYPE_YOLOV5_RAW,   // [1,25200,6] YOLOv5 原始输出 (yolov5s_ball_640_yuv)
    SUBTYPE_YOLO26_MULTI, // 3 头输出（P3/P4/P5），需合并
};

// ---- 预处理管线 ----
enum PreprocPipeline {
    PREPROC_YUV_PAD32,   // YUV420SP via DVPP + AIPP pad32
    PREPROC_RGB_640,     // RGB 640×640, 无 AIPP (ball_aipp.om)
    PREPROC_RGB_LETTERBOX, // RGB letterbox 640×640
};

// ---- 模型描述 ----
typedef struct {
    const char* model_path;
    int   device_id;
    // 输入
    int   input_width;    // 模型输入宽 (640)
    int   input_height;   // 模型输入高 (576 for YUV pad32, 640 for RGB)
    int   display_width;  // 显示/相机宽 (800)
    int   display_height; // 显示/相机高 (600)
    // 预处理
    PreprocPipeline preproc;
    int   aiip_pad_top;   // AIPP top padding (32 for YUV pad32)
    // 坐标映射 (旧版 mx_infer 用 src_w=832, src_h=608)
    float scale_sx;       // sx = src_w / 640.0f
    float scale_sy;       // sy = src_h / 576.0f
    int   pad_top;        // AIPP 顶部 padding (32)
    // 后处理
    ModelSubtype  subtype;
    float conf_threshold;
    float nms_threshold;
    int   num_classes;
} ModelConfig;

// ---- Opaque context ----
typedef struct Mx2Ctx Mx2Ctx;

// ---- Per-stage timing ----
typedef struct {
    long decode_us;
    long resize_us;
    long infer_us;
    long post_us;
    int  content_w;
    int  pad_top;
} Mx2Timing;

// ---- API ----
Mx2Ctx*  mx2_init(const ModelConfig* cfg);
void     mx2_cleanup(Mx2Ctx* ctx);

// 推理: 输入 JPEG 字节，输出检测结果数组
//   res[0] = 检测数量 n
//   res[1..n*6] = [cls_id, x1, y1, x2, y2, conf]
// 坐标已映射到 display 空间。
float*   mx2_infer(Mx2Ctx* ctx, uint8_t* jpeg_data, int jpeg_size,
                    Mx2Timing* timing);

int      mx2_output_count(Mx2Ctx* ctx);
void     mx2_output_shape(Mx2Ctx* ctx, int idx,
                          unsigned long* d0, unsigned long* d1, unsigned long* d2);

#ifdef __cplusplus
}
#endif