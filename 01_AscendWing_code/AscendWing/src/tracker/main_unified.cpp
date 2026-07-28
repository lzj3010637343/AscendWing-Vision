/**
 * main_unified.cpp — AscendWing unified tracker (C++, YOLO only)
 *
 * Camera → NPU inference → AShmSyncFrame (dets + raw JPEG) → /sync_frame_shm
 *
 * SOT tracking + display → sot_live.py (Python, reads sync_frame_shm)
 */
#include "mx_wrapper.h"
#include "sync_frame_shm.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <fstream>
#include <csignal>
#include <chrono>
#include <vector>

static volatile bool g_run = true;
void sigint_handler(int) { g_run = false; }

int main(int argc, char** argv) {
    const char* model_path = "./models/yolo26s_yuv_pad32_topk.om";
    if (argc > 1 && argv[1][0] != '-') model_path = argv[1];

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // Init mxVision
    MxCtx* ctx = mx_init(model_path, 0);
    if (!ctx) { fprintf(stderr, "mx_init failed\n"); return 1; }
    fprintf(stderr, "[UNI] Model: %s\n", model_path);

    // Init sync frame writer (→ sot_live.py)
    if (syncFrameWriterInit() != SYNC_OK) {
        fprintf(stderr, "[UNI] sync frame shm failed\n");
        return 1;
    }
    fprintf(stderr, "[UNI] Sync frame shm ready (%zu bytes)\n", sizeof(AShmSyncFrame));

    // Camera setup
    system("v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=70 -c gain=40 "
           "-c sharpness=0 -c contrast=5 -c gamma=90 2>/dev/null &");
    cv::VideoCapture cap;
    bool cam_ok = false;
    for (int i = 0; i <= 3; i++) {
        cap.open(i, cv::CAP_V4L2);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 800);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
            cap.set(cv::CAP_PROP_FPS, 90);
            cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 2);
            cv::Mat t;
            if (cap.read(t) && !t.empty()) { cam_ok = true; break; }
            cap.release();
        }
    }

    std::vector<uint8_t> file_jpg;
    int src_w = 832, src_h = 608;
    int content_w = 800, content_h = 600;
    if (!cam_ok) {
        std::ifstream ifs("/root/yolo26/test.jpg", std::ios::binary);
        file_jpg.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        fprintf(stderr, "[UNI] No camera, using file: %zu bytes\n", file_jpg.size());
    }

    int frame = 0;
    using namespace std::chrono;
    long total_cap = 0, total_inf = 0;
    long total_decode = 0, total_resize = 0, total_npu = 0, total_topk = 0;

    fprintf(stderr, "[UNI] Running (YOLO → sync shm, SOT+display on sot_live.py)...\n");

    // BENCH=N: per-frame CSV timing to stderr, auto-stop after N frames (0=off).
    static int bench = -1;
    if (bench < 0) bench = getenv("BENCH") ? atoi(getenv("BENCH")) : 0;

    while (g_run) {
        auto t0 = high_resolution_clock::now();

        // ---- Capture JPEG ----
        // 直接用 cv::Mat 的连续 buffer, 避免 cam_buf.assign 多一次拷贝.
        //   依据: MJPG 解码的 Mat 通常连续; 加 isContinuous() 保护, 不连续才回退 assign.
        uint8_t* jpeg_ptr; int jpeg_size;
        cv::Mat mjpeg_frame;
        std::vector<uint8_t> cam_buf;  // 仅不连续时回退使用
        if (cam_ok) {
            if (!cap.read(mjpeg_frame) || mjpeg_frame.empty()) continue;
            if (mjpeg_frame.isContinuous()) {
                jpeg_ptr = mjpeg_frame.data;
                jpeg_size = (int)(mjpeg_frame.total() * mjpeg_frame.elemSize());
            } else {
                cam_buf.assign(mjpeg_frame.data, mjpeg_frame.data + mjpeg_frame.total() * mjpeg_frame.elemSize());
                jpeg_ptr = cam_buf.data(); jpeg_size = (int)cam_buf.size();
            }
        } else {
            jpeg_ptr = file_jpg.data(); jpeg_size = (int)file_jpg.size();
        }

        // ---- NPU Inference ----
        MxTiming tim = {};
        uint8_t* yuv_ptr = nullptr;
        float ratio = 1.0f;
        // need_yuv_host=false: unified 不消费 YUV 数据, 跳过 720KB D2H+紧凑化, 省 ~1-2ms/帧.
        //   解码失败仍可由 (yuv_ptr==nullptr && decode_us==0) 检测.
        float* res = mx_infer(ctx, jpeg_ptr, jpeg_size, &yuv_ptr, &src_w, &src_h,
                              &ratio, &content_w, &content_h, nullptr, &tim, /*need_yuv_host=*/false);
        int n = (int)res[0];

        if (yuv_ptr == nullptr && tim.decode_us == 0) continue;
        if (content_w <= 0) content_w = tim.content_w > 0 ? tim.content_w : src_w;
        if (content_h <= 0) content_h = src_h;
        auto t1 = high_resolution_clock::now();

        // Coordinate mapping
        float sx = (float)src_w / 640.0f;
        float sy = (float)src_h / 576.0f;
        int pad_top = tim.pad_top;
        static int once = 0;
        if (!once++) fprintf(stderr, "[UNI] buf=%dx%d cw=%d pad=%d\n",
                              src_w, src_h, content_w, pad_top);

        // ---- Write sync frame (dets + raw JPEG → sot_live.py) ----
        AShmSyncFrame sync;
        memset(&sync, 0, sizeof(sync));
        sync.frame_id      = (uint32_t)frame;
        sync.timestamp_ns  = (uint64_t)duration_cast<nanoseconds>(
            high_resolution_clock::now().time_since_epoch()).count();
        sync.num_detections = (uint32_t)n;
        sync.jpeg_size     = (uint32_t)jpeg_size;

        for (int i = 0; i < n && i < ASHM_MAX_DETECTIONS; i++) {
            int off = 1 + i * 6;
            int x1 = (int)(res[off+1] * sx);
            int y1 = (int)((res[off+2] - pad_top) * sy);
            int x2 = (int)(res[off+3] * sx);
            int y2 = (int)((res[off+4] - pad_top) * sy);
            if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
            if (x2 <= x1 || y2 <= y1) continue;
            sync.detections[i].class_id   = (int32_t)res[off];
            sync.detections[i].confidence = res[off+5];
            sync.detections[i].x          = (float)x1;
            sync.detections[i].y          = (float)y1;
            sync.detections[i].width      = (float)(x2 - x1);
            sync.detections[i].height     = (float)(y2 - y1);
        }
        memcpy(sync.jpeg_data, jpeg_ptr, jpeg_size);
        syncFramePut(&sync);

        auto t2 = high_resolution_clock::now();

        // Timing
        total_cap += duration_cast<microseconds>(t1 - t0).count();
        total_inf += duration_cast<microseconds>(t2 - t1).count();
        total_decode += tim.decode_us;
        total_resize += tim.resize_us;
        total_npu    += tim.infer_us;
        total_topk   += tim.topk_us;

        if (bench) {
            long mx_us = tim.decode_us + tim.resize_us + tim.infer_us + tim.topk_us;
            long cap_us = (long)duration_cast<microseconds>(t1 - t0).count() - mx_us;
            long wrt_us = (long)duration_cast<microseconds>(t2 - t1).count();
            fprintf(stderr, "[BENCH] f=%u cap=%ld dec=%ld res=%ld npu=%ld top=%ld wrt=%ld tot=%ld ts=%llu\n",
                    sync.frame_id, cap_us, tim.decode_us, tim.resize_us, tim.infer_us, tim.topk_us, wrt_us,
                    cap_us + mx_us + wrt_us, (unsigned long long)sync.timestamp_ns);
            fflush(stderr);
            if (bench > 0 && (int)frame >= bench) g_run = 0;
        }

        if (++frame % 30 == 0) {
            float N = 30.0f;
            fprintf(stderr, "\r\033[K[UNI] FPS:%.1f D:%.1f R:%.1f N:%.1f K:%.1fms | Dets:%d",
                    1000.0f / ((total_cap + total_inf) / N / 1000.0f),
                    total_decode/N/1000, total_resize/N/1000, total_npu/N/1000, total_topk/N/1000, n);
            fflush(stderr);
            total_cap = total_inf = 0;
            total_decode = total_resize = total_npu = total_topk = 0;
        }
    }

    syncFrameWriterCleanup();
    if (cam_ok) cap.release();
    mx_cleanup(ctx);
    system("v4l2-ctl -d /dev/video0 -c exposure_auto=3 2>/dev/null &");
    fprintf(stderr, "\n[UNI] Done\n");
    return 0;
}
