/**
 * mx_wrapper_v2.cpp — Multi-model inference wrapper for Ascend 310B.
 *
 * Supports three model subtypes via ModelConfig:
 *   SUBTYPE_TOPK         — [1,300,6] TopK output (existing YUV pad32 models)
 *   SUBTYPE_YOLOV5_RAW   — [1,25200,6] raw YOLOv5 output
 *   SUBTYPE_YOLO26_MULTI — 3-head raw output (P3/P4/P5)
 *
 * Preproc pipelines:
 *   PREPROC_YUV_PAD32     — DVPP Decode → CropResize → AIPP
 *   PREPROC_RGB_640       — OpenCV BGR→RGB→640×640
 *   PREPROC_RGB_LETTERBOX  — OpenCV BGR→RGB→letterbox→640×640
 */

#define _GLIBCXX_USE_CXX11_ABI 0
#include "mx_model.h"

#include <MxBase/MxBase.h>
#include <MxBase/E2eInfer/ImageProcessor/ImageProcessor.h>
#include <MxBase/E2eInfer/Model/Model.h>
#include <MxBase/E2eInfer/Image/Image.h>
#include <MxBase/E2eInfer/DataType.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <vector>
#include <chrono>

// ---- FP16 → FP32 ----
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)((h >> 15) & 1) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        f = sign;
    } else if (exp == 0x1f) {
        f = sign | (0xff << 23) | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r; memcpy(&r, &f, sizeof(r)); return r;
}

// ---- NMS ----
static int apply_nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores,
                      std::vector<int>& cls_ids, float nms_thresh,
                      float* out_buf, int max_det) {
    if (boxes.empty()) { out_buf[0] = 0; return 0; }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, nms_thresh, keep);
    int m = 0;
    for (int idx : keep) {
        if (m >= max_det) break;
        int o = 1 + m * 6;
        out_buf[o+0] = (float)cls_ids[idx];
        out_buf[o+1] = (float)boxes[idx].x;
        out_buf[o+2] = (float)boxes[idx].y;
        out_buf[o+3] = (float)(boxes[idx].x + boxes[idx].width);
        out_buf[o+4] = (float)(boxes[idx].y + boxes[idx].height);
        out_buf[o+5] = scores[idx];
        m++;
    }
    out_buf[0] = (float)m;
    return m;
}

// ---- Internal context ----
struct Mx2CtxImpl {
    ModelConfig cfg;
    std::shared_ptr<MxBase::ImageProcessor> proc;
    std::shared_ptr<MxBase::Model> model;
    int output_count = 0;
    unsigned long shapes[4][3] = {};
    bool is_fp16 = true;            // FP16 model → need conversion
    float res_buf[1 + 300 * 6] = {}; // max 300 detections
    std::vector<uint8_t> yuv_buf;
};

// ---- Init ----
Mx2Ctx* mx2_init(const ModelConfig* cfg) {
    if (!cfg) return nullptr;
    if (MxBase::MxInit() != APP_ERR_OK) return nullptr;

    auto ctx = new Mx2CtxImpl();
    ctx->cfg = *cfg;

    // Load model
    std::string mp(cfg->model_path);
    ctx->model = std::make_shared<MxBase::Model>(mp, cfg->device_id);
    if (!ctx->model) { delete ctx; return nullptr; }

    // Detect output format
    ctx->output_count = ctx->model->GetOutputTensorNum();
    for (int i = 0; i < ctx->output_count && i < 4; i++) {
        auto s = ctx->model->GetOutputTensorShape(i);
        for (int j = 0; j < 3 && j < (int)s.size(); j++)
            ctx->shapes[i][j] = s[j];
    }

    // Init preprocessor
    if (cfg->preproc == PREPROC_YUV_PAD32) {
        ctx->proc = std::make_shared<MxBase::ImageProcessor>(cfg->device_id);
    }

    fprintf(stderr, "[MX2] Loaded %s  subtype=%d  outputs=%d  preproc=%d\n",
            cfg->model_path, cfg->subtype, ctx->output_count, cfg->preproc);
    for (int i = 0; i < ctx->output_count; i++)
        fprintf(stderr, "[MX2]   out[%d]: %lux%lux%lu\n", i,
                ctx->shapes[i][0], ctx->shapes[i][1], ctx->shapes[i][2]);

    return (Mx2Ctx*)ctx;
}

void mx2_cleanup(Mx2Ctx* ctx) {
    auto* c = (Mx2CtxImpl*)ctx;
    if (!c) return;
    c->model.reset();
    c->proc.reset();
    delete c;
    MxBase::MxDeInit();
}

// ---- INFER ----
float* mx2_infer(Mx2Ctx* _ctx, uint8_t* jpeg_data, int jpeg_size,
                 Mx2Timing* timing) {
    auto* ctx = (Mx2CtxImpl*)_ctx;
    ctx->res_buf[0] = 0;
    if (timing) memset(timing, 0, sizeof(Mx2Timing));

    if (!ctx || !ctx->model) return ctx->res_buf;

    MxBase::Image infer_img;
    int pad_top = ctx->cfg.aiip_pad_top;

    // ===== Stage 1: Preprocessing =====
    auto t0 = std::chrono::high_resolution_clock::now();

    if (ctx->cfg.preproc == PREPROC_YUV_PAD32) {
        // DVPP: Decode JPEG → YUV420SP
        std::shared_ptr<uint8_t> data(jpeg_data, [](uint8_t*){});
        MxBase::Image yuv;
        if (ctx->proc->Decode(data, jpeg_size, yuv,
                              MxBase::ImageFormat::YUV_SP_420) != APP_ERR_OK)
            return ctx->res_buf;

        int buf_w = yuv.GetSize().width;
        int buf_h = yuv.GetSize().height;
        int orig_w = yuv.GetOriginalSize().width;
        int orig_h = yuv.GetOriginalSize().height;
        if (orig_w <= 0) orig_w = buf_w; if (orig_h <= 0) orig_h = buf_h;

        // CropResize
        if (buf_w > orig_w) {
            yuv.SetImageOriginalSize(MxBase::Size(buf_w, buf_h));
            std::vector<MxBase::Rect> crops = {
                MxBase::Rect(0, 0, orig_w == 800 ? 800 : (float)orig_w, (float)buf_h)};
            std::vector<MxBase::Image> crop_results(1);
            ctx->proc->CropResize(yuv, crops,
                MxBase::Size(ctx->cfg.input_width, ctx->cfg.input_height),
                crop_results);
            infer_img = crop_results[0];
        } else {
            ctx->proc->Resize(yuv,
                MxBase::Size(ctx->cfg.input_width, ctx->cfg.input_height), infer_img);
        }

        if (timing) {
            timing->content_w = orig_w;
            timing->pad_top   = pad_top;
        }
    }
    else if (ctx->cfg.preproc == PREPROC_RGB_640 || ctx->cfg.preproc == PREPROC_RGB_LETTERBOX) {
        // OpenCV decode → BGR → RGB → 640×640
        cv::Mat raw(1, jpeg_size, CV_8UC1, jpeg_data);
        cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
        if (bgr.empty()) return ctx->res_buf;

        int src_w = bgr.cols, src_h = bgr.rows;
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        if (ctx->cfg.preproc == PREPROC_RGB_LETTERBOX) {
            float scale = std::min(640.0f / src_w, 640.0f / src_h);
            int nw = (int)(src_w * scale), nh = (int)(src_h * scale);
            int pl = (640 - nw) / 2, pt = (640 - nh) / 2;
            cv::Mat lb = cv::Mat::zeros(640, 640, CV_8UC3);
            cv::resize(rgb, resized, cv::Size(nw, nh));
            resized.copyTo(lb(cv::Rect(pl, pt, nw, nh)));
            rgb = lb;
        } else {
            cv::resize(rgb, rgb, cv::Size(640, 640));
        }

        auto sp = std::make_shared<std::vector<uint8_t>>(
            rgb.total() * rgb.elemSize());
        memcpy(sp->data(), rgb.data, sp->size());
        infer_img = MxBase::Image(
            std::shared_ptr<uint8_t>(sp, sp->data()), sp->size(), -1,
            MxBase::Size(640, 640), MxBase::ImageFormat::RGB_888);

        if (timing) {
            timing->content_w = src_w;
            timing->pad_top   = 0;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // ===== Stage 2: NPU Inference =====
    std::vector<MxBase::Tensor> inputs = { infer_img.ConvertToTensor() };
    auto outputs = ctx->model->Infer(inputs);
    if (outputs.size() < 1) return ctx->res_buf;

    auto t2 = std::chrono::high_resolution_clock::now();

    // ===== Stage 3: Post-processing =====
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> cls_ids;
    float sx = ctx->cfg.scale_sx;
    float sy = ctx->cfg.scale_sy;

    if (ctx->cfg.subtype == SUBTYPE_TOPK) {
        // [1,300,6] TopK — FP16 or FP32
        outputs[0].ToHost();
        void* raw = outputs[0].GetData();
        int nrow = (int)ctx->shapes[0][1];

        for (int i = 0; i < nrow; i++) {
            float x1,y1,x2,y2,cid,cf;
            if (ctx->is_fp16) {
                uint16_t* d16 = (uint16_t*)raw;
                x1  = fp16_to_f32(d16[i*6]);
                y1  = fp16_to_f32(d16[i*6+1]);
                x2  = fp16_to_f32(d16[i*6+2]);
                y2  = fp16_to_f32(d16[i*6+3]);
                cid = fp16_to_f32(d16[i*6+4]);
                cf  = fp16_to_f32(d16[i*6+5]);
            } else {
                float* d32 = (float*)raw;
                x1  = d32[i*6];
                y1  = d32[i*6+1];
                x2  = d32[i*6+2];
                y2  = d32[i*6+3];
                cid = d32[i*6+4];
                cf  = d32[i*6+5];
            }
            if (cf < ctx->cfg.conf_threshold) continue;

            int ix1 = (int)(x1 * sx);
            int iy1 = (int)((y1 - pad_top) * sy);
            int ix2 = (int)(x2 * sx);
            int iy2 = (int)((y2 - pad_top) * sy);
            if (ix1 < 0) ix1 = 0; if (iy1 < 0) iy1 = 0;
            if (ix2 <= ix1 || iy2 <= iy1) continue;

            boxes.push_back(cv::Rect(ix1, iy1, ix2-ix1, iy2-iy1));
            scores.push_back(cf);
            cls_ids.push_back((int)cid);
        }
    }
    else if (ctx->cfg.subtype == SUBTYPE_YOLOV5_RAW) {
        // [1,25200, N] YOLOv5 raw
        outputs[0].ToHost();
        float* out = (float*)outputs[0].GetData();
        int nrow = (int)ctx->shapes[0][1];
        int ncol = (int)ctx->shapes[0][2]; // 5 + num_classes

        for (int i = 0; i < nrow; i++) {
            float obj = out[i * ncol + 4];
            if (obj < ctx->cfg.conf_threshold) continue;
            int bc = 0; float bs = 0;
            for (int c = 5; c < ncol; c++) {
                if (out[i * ncol + c] > bs) { bs = out[i * ncol + c]; bc = c-5; }
            }
            float cx = out[i*ncol], cy = out[i*ncol+1], w = out[i*ncol+2], h = out[i*ncol+3];
            int x1 = (int)((cx - w/2) * sx), y1 = (int)((cy - h/2) * sy);
            int x2 = (int)((cx + w/2) * sx), y2 = (int)((cy + h/2) * sy);
            if (x1<0) x1=0; if (y1<0) y1=0; if (x2<=x1||y2<=y1) continue;
            boxes.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
            scores.push_back(obj * bs);
            cls_ids.push_back(bc);
        }
    }
    else if (ctx->cfg.subtype == SUBTYPE_YOLO26_MULTI) {
        // Multi-head: read all output tensors, each [1, N_i, 4+num_classes]
        for (int h = 0; h < ctx->output_count; h++) {
            outputs[h].ToHost();
            float* out = (float*)outputs[h].GetData();
            int nrow = (int)ctx->shapes[h][1];
            int ncol = (int)ctx->shapes[h][2]; // 4 + num_classes

            for (int i = 0; i < nrow; i++) {
                int bc = 0; float bs = 0;
                float cx = out[i*ncol], cy = out[i*ncol+1];
                float w = out[i*ncol+2], h = out[i*ncol+3];

                // Find best class score
                for (int c = 4; c < ncol; c++) {
                    if (out[i*ncol + c] > bs) { bs = out[i*ncol + c]; bc = c-4; }
                }
                if (bs < ctx->cfg.conf_threshold) continue;

                int x1 = (int)((cx - w/2) * sx), y1 = (int)((cy - h/2) * sy);
                int x2 = (int)((cx + w/2) * sx), y2 = (int)((cy + h/2) * sy);
                if (x1<0) x1=0; if (y1<0) y1=0; if (x2<=x1||y2<=y1) continue;
                boxes.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                scores.push_back(bs);
                cls_ids.push_back(bc);
            }
        }
    }

    // NMS
    apply_nms(boxes, scores, cls_ids,
                       ctx->cfg.nms_threshold, ctx->res_buf, 300);

    auto t3 = std::chrono::high_resolution_clock::now();
    if (timing) {
        using namespace std::chrono;
        timing->decode_us = duration_cast<microseconds>(t1 - t0).count();
        timing->resize_us = 0;
        timing->infer_us  = duration_cast<microseconds>(t2 - t1).count();
        timing->post_us   = duration_cast<microseconds>(t3 - t2).count();
    }

    return ctx->res_buf;
}

int mx2_output_count(Mx2Ctx* _ctx) {
    auto* ctx = (Mx2CtxImpl*)_ctx;
    return ctx ? ctx->output_count : 0;
}

void mx2_output_shape(Mx2Ctx* _ctx, int idx, unsigned long* d0, unsigned long* d1, unsigned long* d2) {
    auto* ctx = (Mx2CtxImpl*)_ctx;
    if (ctx && idx >= 0 && idx < ctx->output_count) {
        if (d0) *d0 = ctx->shapes[idx][0];
        if (d1) *d1 = ctx->shapes[idx][1];
        if (d2) *d2 = ctx->shapes[idx][2];
    }
}