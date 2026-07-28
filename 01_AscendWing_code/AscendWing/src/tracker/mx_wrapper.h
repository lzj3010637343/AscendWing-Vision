#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context
typedef struct MxCtx MxCtx;

// Per-stage timing in microseconds
typedef struct {
    long decode_us;   // JPEG→YUV decode (DVPP/ImageProcessor)
    long resize_us;   // YUV resize to 640x576
    long infer_us;    // NPU model inference
    long topk_us;     // CPU TopK decode (bbox+cls → detections)
    int  content_w;   // display/content width (for crop)
    int  pad_top;     // AIPP top padding offset
} MxTiming;

MxCtx*  mx_init(const char* model_path, int device_id);
void    mx_cleanup(MxCtx* ctx);

// Run inference on JPEG bytes. Returns float buffer [count, cls, x1,y1,x2,y2, conf, ...]
// out_yuv: decoded YUV NV12 data pointer (valid until next call)
// out_w, out_h: decoded image dimensions
// out_ratio, out_dw, out_dh: letterbox params for coordinate mapping
//   x_orig = (x_model - dw) / ratio,  y_orig = (y_model - dh) / ratio
// out_timing: if non-null, filled with per-stage microseconds
// need_yuv_host: 若 false, 跳过 YUV 的 D2H 拷贝与紧凑化(用于不消费 YUV 的路径, 省 ~1-2ms/帧).
//   注意: need_yuv_host=false 时 out_yuv 仍为 nullptr, 调用方可据 (out_yuv==nullptr && decode_us==0) 判解码失败.
float*  mx_infer(MxCtx* ctx, uint8_t* jpeg_data, int size,
                 uint8_t** out_yuv, int* out_w, int* out_h,
                 float* out_ratio, int* out_dw, int* out_dh,
                 int* out_yuv_stride, MxTiming* out_timing,
                 bool need_yuv_host = true);

// Get model output tensor count (debug)
int     mx_output_count(MxCtx* ctx);
void    mx_output_shape(MxCtx* ctx, int idx, unsigned long* d0, unsigned long* d1, unsigned long* d2);

#ifdef __cplusplus
}
#endif
