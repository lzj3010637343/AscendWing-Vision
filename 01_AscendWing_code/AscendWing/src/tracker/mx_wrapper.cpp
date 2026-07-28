#define _GLIBCXX_USE_CXX11_ABI 0
// #define ENABLE_AI_CORE_DEBUG  // uncomment + add ascendcl link for AI Core exception callback

#include "mx_wrapper.h"
#include <MxBase/MxBase.h>
#include <MxBase/E2eInfer/ImageProcessor/ImageProcessor.h>
#include <MxBase/E2eInfer/Model/Model.h>
#include <MxBase/E2eInfer/Tensor/Tensor.h>
#ifdef ENABLE_AI_CORE_DEBUG
#include "acl/acl.h"
#endif
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>

// ---- FP16 → FP32 conversion (IEEE 754 standard, verified) ----
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)((h >> 15) & 1) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f;
    if (exp == 0) {
        // Zero / subnormal — flush to zero for simplicity (NPU rarely outputs subnormals)
        f = sign;
    } else if (exp == 0x1f) {
        // Inf / NaN
        f = sign | (0xff << 23) | (mant << 13);
    } else {
        // Normalized: exp_bias=15 → FP32=127
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

#ifdef ENABLE_AI_CORE_DEBUG
// ==================== AI Core 异常回调 ====================
static void ai_core_exception_callback(aclrtExceptionInfo* info) {
    uint32_t dev_id = aclrtGetDeviceIdFromExceptionInfo(info);
    uint32_t stream_id = aclrtGetStreamIdFromExceptionInfo(info);
    uint32_t task_id = aclrtGetTaskIdFromExceptionInfo(info);

    char op_name[256] = {0};
    aclTensorDesc* input_desc = nullptr;
    aclTensorDesc* output_desc = nullptr;
    size_t input_cnt = 0, output_cnt = 0;

    aclError ret = aclmdlCreateAndGetOpDesc(dev_id, stream_id, task_id,
                                             op_name, sizeof(op_name),
                                             &input_desc, &input_cnt,
                                             &output_desc, &output_cnt);
    fprintf(stderr, "\n\033[1;31m[AI_CORE_ERR]\033[0m device=%u stream=%u task=%u op=%s\n",
            dev_id, stream_id, task_id, ret == ACL_SUCCESS ? op_name : "(unknown)");

    if (ret == ACL_SUCCESS) {
        for (size_t i = 0; i < input_cnt; i++) {
            const aclTensorDesc* desc = aclGetTensorDescByIndex(input_desc, i);
            if (!desc) continue;
            void* addr = aclGetTensorDescAddress(desc);
            auto fmt = aclGetTensorDescFormat(desc);
            auto dtype = aclGetTensorDescType(desc);
            auto ndim = aclGetTensorDescNumDims(desc);
            fprintf(stderr, "  input[%zu]: addr=%p fmt=%d dtype=%d dims=[", i, addr, (int)fmt, (int)dtype);
            for (size_t d = 0; d < ndim; d++) {
                int64_t dim = 0;
                aclGetTensorDescDimV2(desc, d, &dim);
                fprintf(stderr, "%s%lld", d ? "," : "", (long long)dim);
            }
            fprintf(stderr, "]\n");
        }
        for (size_t i = 0; i < output_cnt; i++) {
            const aclTensorDesc* desc = aclGetTensorDescByIndex(output_desc, i);
            if (!desc) continue;
            void* addr = aclGetTensorDescAddress(desc);
            auto fmt = aclGetTensorDescFormat(desc);
            auto dtype = aclGetTensorDescType(desc);
            auto ndim = aclGetTensorDescNumDims(desc);
            fprintf(stderr, "  output[%zu]: addr=%p fmt=%d dtype=%d dims=[", i, addr, (int)fmt, (int)dtype);
            for (size_t d = 0; d < ndim; d++) {
                int64_t dim = 0;
                aclGetTensorDescDimV2(desc, d, &dim);
                fprintf(stderr, "%s%lld", d ? "," : "", (long long)dim);
            }
            fprintf(stderr, "]\n");
        }
        aclDestroyTensorDesc(input_desc);
        aclDestroyTensorDesc(output_desc);
    }

    // 获取并打印错误码
    fprintf(stderr, "  Check /var/log/npu/slog for full dump\n");
    fflush(stderr);
}
#endif  // ENABLE_AI_CORE_DEBUG

struct MxCtx {
    std::shared_ptr<MxBase::ImageProcessor> proc;
    std::shared_ptr<MxBase::Model> model;
    float res_buf[1 + 300 * 6];
    std::vector<uint8_t> yuv_buf;
    int output_count = 0;
    unsigned long shapes[3][3] = {};
    int device_id = 0;
    // 预分配的输出 tensor, 每帧复用 (替代 1参 Infer 每帧内部分配).
    //   依据: Model.h 3参 Infer 文档 "Strongly recommend Tensor.Malloc() to assign memory for output tensors"
    std::vector<MxBase::Tensor> out_tensors;
};

static const int NA = 8400, NC = 10, TK = 300;

static void cpu_decode(const float* bbox, const float* cls, float* res) {
    struct AS { float s; int a, c; };
    std::vector<AS> scores(NA);
    for (int a = 0; a < NA; a++) {
        float best = -1; int bc = 0;
        for (int c = 0; c < NC; c++)
            if (cls[a * NC + c] > best) { best = cls[a * NC + c]; bc = c; }
        scores[a] = {best, a, bc};
    }
    int n = std::min(TK, NA);
    std::partial_sort(scores.begin(), scores.begin() + n, scores.end(),
        [](const AS& x, const AS& y) { return x.s > y.s; });
    int cnt = 0;
    for (int i = 0; i < n && scores[i].s > 0.25f; i++) {
        int a = scores[i].a, o = 1 + cnt * 6;
        const float* b = bbox + a * 4;
        res[o+0] = (float)scores[i].c;
        res[o+1] = b[0]; res[o+2] = b[1]; res[o+3] = b[2]; res[o+4] = b[3];
        res[o+5] = scores[i].s; cnt++;
    }
    res[0] = (float)cnt;
}

MxCtx* mx_init(const char* model_path, int device_id) {
    if (MxBase::MxInit() != APP_ERR_OK) return nullptr;
#ifdef ENABLE_AI_CORE_DEBUG
    // Register AI Core exception callback (once per process)
    static bool cb_registered = false;
    if (!cb_registered) {
        aclrtSetExceptionInfoCallback(ai_core_exception_callback);
        cb_registered = true;
    }
#endif
    auto ctx = new MxCtx();
    ctx->device_id = device_id;
    ctx->proc = std::make_shared<MxBase::ImageProcessor>(device_id);
    std::string mp(model_path);
    ctx->model = std::make_shared<MxBase::Model>(mp, device_id);
    if (!ctx->model) { delete ctx; return nullptr; }

    ctx->output_count = ctx->model->GetOutputTensorNum();
    for (int i = 0; i < ctx->output_count && i < 3; i++) {
        auto s = ctx->model->GetOutputTensorShape(i);
        for (int j = 0; j < 3 && j < (int)s.size(); j++)
            ctx->shapes[i][j] = s[j];
    }
    // 预分配输出 tensor 复用: 用模型实际输出 shape/dtype 构造 + Malloc (Model.h 文档要求).
    //   每帧 Infer 复用同一块 device 内存, 省每帧 allocator 开销 + 减少碎片.
    for (int i = 0; i < ctx->output_count; i++) {
        auto s = ctx->model->GetOutputTensorShape(i);          // vector<uint32_t>
        auto dt = ctx->model->GetOutputTensorDataType(i);      // TensorDType
        MxBase::Tensor t(s, dt, device_id);
        if (t.Malloc() != APP_ERR_OK) {
            fprintf(stderr, "[ERR] output tensor[%d] Malloc failed\n", i);
            delete ctx; return nullptr;
        }
        ctx->out_tensors.push_back(std::move(t));
    }
    // 输出 dtype 校验: mx_infer 把 outputs[0] 当 FP32 读 (float* cast, mx_wrapper.cpp:224).
    //   若实际为 FP16 会读到垃圾值. 此处仅打印, 不改逻辑, 便于发现隐患.
    //   依据: Model.h:109 GetOutputTensorDataType, DataType.h TensorDType (FLOAT32=0, FLOAT16=1)
    static int once_dtype = 0;
    if (!once_dtype++) {
        auto dt = ctx->model->GetOutputTensorDataType(0);
        fprintf(stderr, "[MX] output[0] dtype=%d (0=FP32,1=FP16) shape=[%lu,%lu,%lu]\n",
                (int)dt,
                (unsigned long)ctx->shapes[0][0], (unsigned long)ctx->shapes[0][1], (unsigned long)ctx->shapes[0][2]);
    }
    return ctx;
}

void mx_cleanup(MxCtx* ctx) {
    if (!ctx) return;
    ctx->model.reset();
    ctx->proc.reset();
    delete ctx;
    MxBase::MxDeInit();
}

float* mx_infer(MxCtx* ctx, uint8_t* jpeg_data, int size,
                uint8_t** out_yuv, int* out_w, int* out_h,
                float* out_ratio, int* out_dw, int* out_dh,
                int* out_yuv_stride, MxTiming* out_timing,
                bool need_yuv_host) {
    ctx->res_buf[0] = 0;
    if (out_yuv) *out_yuv = nullptr;
    if (out_ratio) *out_ratio = 1.0f;
    if (out_dw) *out_dw = 0;
    if (out_dh) *out_dh = 0;
    if (out_yuv_stride) *out_yuv_stride = 0;
    if (out_timing) { memset(out_timing, 0, sizeof(MxTiming)); out_timing->pad_top = 32; }

    if (!ctx || !ctx->model) return ctx->res_buf;

    // Stage 1: JPEG decode
    auto t0 = std::chrono::high_resolution_clock::now();
    std::shared_ptr<uint8_t> data(jpeg_data, [](uint8_t*){});
    MxBase::Image yuv;
    if (ctx->proc->Decode(data, size, yuv, MxBase::ImageFormat::YUV_SP_420) != APP_ERR_OK)
        return ctx->res_buf;

    int buf_w = yuv.GetSize().width;    // aligned (832)
    int buf_h = yuv.GetSize().height;   // aligned (608)
    int orig_w = yuv.GetOriginalSize().width;   // content (800)
    int orig_h = yuv.GetOriginalSize().height;  // content (600)
    static int once2 = 0;
    if (!once2++) fprintf(stderr, "[DVPP] buf=%dx%d  original=%dx%d\n", buf_w, buf_h, orig_w, orig_h);

    // Stage 2: Crop right 32px, keep bottom 8px, resize to 640x576
    auto tm = std::chrono::high_resolution_clock::now();
    MxBase::Image rz;
    if (orig_w > 0 && orig_h > 0 && buf_w > orig_w) {
        yuv.SetImageOriginalSize(MxBase::Size(buf_w, buf_h));
        std::vector<MxBase::Rect> crops = { MxBase::Rect(0, 0, orig_w, buf_h) };  // 800x608
        std::vector<MxBase::Image> crop_results(1);
        auto ret = ctx->proc->CropResize(yuv, crops, MxBase::Size(640, 576), crop_results);
        static int once_cr = 0;
        if (!once_cr++) fprintf(stderr, "[CROP] ret=%d %dx%d → 640x576 (AIPP pads 32→640)\n", ret, orig_w, buf_h);
        if (ret != APP_ERR_OK) return ctx->res_buf;
        rz = crop_results[0];
        buf_w = orig_w;  // width cropped: 832→800
    } else {
        ctx->proc->Resize(yuv, MxBase::Size(640, 640), rz);
    }

    if (out_w) *out_w = buf_w;
    if (out_h) *out_h = buf_h;
    if (out_dw) *out_dw = orig_w > 0 ? orig_w : buf_w;
    if (out_dh) *out_dh = orig_h > 0 ? orig_h : buf_h;
    auto t1 = std::chrono::high_resolution_clock::now();

    // Stage 3: NPU inference (3参 Infer, 复用预分配的 out_tensors, 省每帧分配)
    std::vector<MxBase::Tensor> inputs = { rz.ConvertToTensor() };
    auto inferRet = ctx->model->Infer(inputs, ctx->out_tensors);
    if (inferRet != APP_ERR_OK || ctx->out_tensors.empty()) return ctx->res_buf;
    auto t2 = std::chrono::high_resolution_clock::now();

    // Stage 4: Copy output (NPU TopK model: [1,300,6], FP32)
    ctx->out_tensors[0].ToHost();
    float* dets = (float*)ctx->out_tensors[0].GetData();
    int n = 0;
    for (int i = 0; i < 300 && dets[i * 6 + 5] > 0.25f; i++) {
        int o = 1 + n * 6;
        ctx->res_buf[o + 0] = dets[i * 6 + 0];
        ctx->res_buf[o + 1] = dets[i * 6 + 1];
        ctx->res_buf[o + 2] = dets[i * 6 + 2];
        ctx->res_buf[o + 3] = dets[i * 6 + 3];
        ctx->res_buf[o + 4] = dets[i * 6 + 4];
        ctx->res_buf[o + 5] = dets[i * 6 + 5];
        n++;
    }
    ctx->res_buf[0] = (float)n;
    auto t3 = std::chrono::high_resolution_clock::now();

    // Copy YUV for display (compact to remove stride)
    // need_yuv_host=false 时跳过: 省去 720KB D2H + ~900 行 memcpy (unified/vision 不消费 YUV)
    // 依据: DVPP D2H 是真实开销; unified/vision 仅用 out_yuv==nullptr 判解码失败, 不读 YUV 数据
    //
    // [临时诊断] 环境变量 AW_FORCE_YUV_HOST=1 强制走 ToHost 路径并计时, 用于实测优化收益.
    //   用法: 同一二进制, 设/不设环境变量各跑一次, 对比 FPS + [BENCH] 打印.
    //   验证完可删.
    static bool g_force_yuv_host = (getenv("AW_FORCE_YUV_HOST") != nullptr);
    static long g_yuv_host_us = 0;
    static int  g_yuv_host_n = 0;
    if (need_yuv_host || g_force_yuv_host) {
        auto th0 = std::chrono::high_resolution_clock::now();
        yuv.ToHost();
        auto imgData = yuv.GetData();
        size_t dataSize = yuv.GetDataSize();
        int yuv_stride = buf_w;
        if (imgData && dataSize > 0) {
            int stride_chk = (int)(dataSize * 2 / (3 * buf_h));
            if (stride_chk > yuv_stride) yuv_stride = stride_chk;
            int cw = orig_w > 0 ? orig_w : buf_w;
            int ch = orig_h > 0 ? orig_h : buf_h;
            size_t compact_sz = (size_t)cw * ch * 3 / 2;
            ctx->yuv_buf.resize(compact_sz);
            const uint8_t* src = imgData.get();
            uint8_t* dst = ctx->yuv_buf.data();
            for (int r = 0; r < ch; r++)
                memcpy(dst + r * cw, src + r * yuv_stride, cw);
            const uint8_t* uv_src = src + yuv_stride * buf_h;
            uint8_t* uv_dst = dst + cw * ch;
            for (int r = 0; r < ch / 2; r++)
                memcpy(uv_dst + r * cw, uv_src + r * yuv_stride, cw);
            if (out_yuv) *out_yuv = ctx->yuv_buf.data();
            if (out_yuv_stride) *out_yuv_stride = cw;
        }
        auto th1 = std::chrono::high_resolution_clock::now();
        g_yuv_host_us += std::chrono::duration_cast<std::chrono::microseconds>(th1 - th0).count();
        g_yuv_host_n++;
        if ((g_yuv_host_n % 30) == 0) {
            fprintf(stderr, "[BENCH] yuv_tohost+compact avg=%.2fms (n=%d)\n",
                    g_yuv_host_us / 1000.0 / g_yuv_host_n, g_yuv_host_n);
        }
    }

    if (out_timing) {
        using namespace std::chrono;
        out_timing->decode_us = duration_cast<microseconds>(tm - t0).count();
        out_timing->resize_us = duration_cast<microseconds>(t1 - tm).count();
        out_timing->infer_us  = duration_cast<microseconds>(t2 - t1).count();
        out_timing->topk_us   = duration_cast<microseconds>(t3 - t2).count();
        out_timing->content_w = orig_w > 0 ? orig_w : buf_w;
        out_timing->pad_top   = 32;
    }

    return ctx->res_buf;
}

int mx_output_count(MxCtx* ctx) { return ctx ? ctx->output_count : 0; }

void mx_output_shape(MxCtx* ctx, int idx, unsigned long* d0, unsigned long* d1, unsigned long* d2) {
    if (!ctx || idx < 0 || idx >= 3) return;
    if (d0) *d0 = ctx->shapes[idx][0];
    if (d1) *d1 = ctx->shapes[idx][1];
    if (d2) *d2 = ctx->shapes[idx][2];
}
