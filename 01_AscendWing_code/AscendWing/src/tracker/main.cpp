#include "mx_wrapper.h"
#include "ascend_shm.h"
#include "sync_frame_shm.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <fstream>
#include <csignal>
#include <chrono>
#include <vector>
#include <algorithm>
#include "mjpeg_server.h"

static volatile bool g_run = true;
void sigint_handler(int) { g_run = false; }

int main(int argc, char** argv) {
    const char* model_path = "./models/yolo26s_yuv_pad32_topk.om";
    if (argc > 1) model_path = argv[1];

    struct sigaction sa = {};
    sa.sa_handler = sigint_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    MxCtx* ctx = mx_init(model_path, 0);
    if (!ctx) { fprintf(stderr, "mx_init failed\n"); return 1; }

    // Initialize POSIX shared memory IPC (free_ws compatible)
    if (aclShmWriterInit() != ASHM_OK)
        fprintf(stderr, "[YOLO26] WARNING: shm writer init failed, IPC disabled\n");
    else
        fprintf(stderr, "[YOLO26] POSIX shm IPC ready (det+img queues)\n");

    // Initialize sync frame IPC (combined detection+JPEG for sot_live.py)
    if (syncFrameWriterInit() != SYNC_OK)
        fprintf(stderr, "[YOLO26] WARNING: sync frame shm init failed\n");
    else
        fprintf(stderr, "[YOLO26] Sync frame shm ready (%zu bytes)\n", sizeof(AShmSyncFrame));

    int no = mx_output_count(ctx);
    fprintf(stderr, "[YOLO26] %s, %d output(s), NPU TopK enabled\n", model_path, no);
    for (int i = 0; i < no; i++) {
        unsigned long d0, d1, d2;
        mx_output_shape(ctx, i, &d0, &d1, &d2);
        fprintf(stderr, "[YOLO26]   out[%u]: [%lu,%lu,%lu]\n", i, d0, d1, d2);
    }

    // Set camera hardware: short exposure + tuned DSP
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
            if (cap.read(t) && !t.empty()) { cam_ok = true; fprintf(stderr, "Cam: /dev/video%d\n", i); break; }
            cap.release();
        }
    }

    std::vector<uint8_t> file_jpg;
    int src_w = 832, src_h = 608;     // default buffer (aligned) size
    int content_w = 800, content_h = 600;  // default content size
    if (!cam_ok) {
        std::ifstream ifs("/root/yolo26/test.jpg", std::ios::binary);
        file_jpg.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        fprintf(stderr, "No camera, using file: %zu bytes\n", file_jpg.size());
        // Get dims by running mx_infer once on the file
        uint8_t* yuv_ptr;
        float ratio;
        MxTiming tim;
        float* r = mx_infer(ctx, file_jpg.data(), file_jpg.size(), &yuv_ptr,
                            &src_w, &src_h, &ratio, &content_w, &content_h, nullptr, &tim);
        fprintf(stderr, "Image: %dx%d\n", src_w, src_h);
    }

    MjpegServer mjpeg(8080);
    mjpeg.start();

    int frame = 0;
    using namespace std::chrono;
    long total_cap = 0, total_inf = 0, total_draw = 0;
    long total_decode = 0, total_resize = 0, total_npu = 0, total_topk = 0;

    while (g_run) {
        // ---- Step 1: Get JPEG bytes ----
        auto t0 = high_resolution_clock::now();
        // 直接用 cv::Mat 的连续 buffer, 避免 cam_buf.assign 多一次拷贝 (不连续才回退).
        // 注意: mjpeg_frame 提到 if 外, 保证 jpeg_ptr 在 mx_infer 期间有效 (Mat 析构会释放 buffer).
        cv::Mat mjpeg_frame;
        std::vector<uint8_t> cam_buf;
        uint8_t* jpeg_ptr; int jpeg_size;
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
        auto t1 = high_resolution_clock::now();

        // ---- Step 2-4: MxBase Decode + Resize + Infer (via wrapper) ----
        MxTiming tim = {};
        uint8_t* yuv_ptr = nullptr;
        float ratio = 1.0f;
        int content_w = 0, content_h = 0;
        float* res = mx_infer(ctx, jpeg_ptr, jpeg_size, &yuv_ptr, &src_w, &src_h,
                              &ratio, &content_w, &content_h, nullptr, &tim);
        int n = (int)res[0];
        auto t2 = high_resolution_clock::now();

        // DVPP decode failed (non-Huffman JPEG, USB glitch) → skip frame
        if (yuv_ptr == nullptr && tim.decode_us == 0) continue;
        if (content_w <= 0) content_w = tim.content_w > 0 ? tim.content_w : src_w;
        if (content_h <= 0) content_h = src_h;
        static int once = 0;
        if (!once++) fprintf(stderr, "[MAP] buf=%dx%d  content_w=%d  pad_top=%d  sx=%.4f sy=%.4f\n",
                              src_w, src_h, content_w, tim.pad_top,
                              (float)src_w/640, (float)src_h/576.0f);

        // ---- Coordinate mapping (model 640x640 → display) ----
        float sx = (float)src_w / 640.0f;
        float sy = (float)src_h / 576.0f;
        int pad_top = tim.pad_top;

        // ---- Write detections to POSIX shm (free_ws IPC, every frame) ----
        {
            AShmDetectionFrame det_frame;
            det_frame.frame_id = frame;
            det_frame.timestamp_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            det_frame.num_detections = 0;
            for (int i = 0; i < n && det_frame.num_detections < ASHM_MAX_DETECTIONS; i++) {
                int off = 1 + i * 6;
                int x1 = (int)(res[off+1] * sx);
                int y1 = (int)((res[off+2] - pad_top) * sy);
                int x2 = (int)(res[off+3] * sx);
                int y2 = (int)((res[off+4] - pad_top) * sy);
                if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
                if (x2 <= x1 || y2 <= y1) continue;
                auto& d = det_frame.detections[det_frame.num_detections++];
                d.class_id   = (int32_t)res[off];
                d.confidence = res[off+5];
                d.x      = (float)x1;
                d.y      = (float)y1;
                d.width  = (float)(x2 - x1);
                d.height = (float)(y2 - y1);
            }
            aclShmPutDetections(&det_frame);

            // ---- Write combined sync frame for multi-process (sot_live.py) ----
            {
                AShmSyncFrame sync;
                sync.frame_id     = (uint32_t)frame;
                sync.timestamp_ns = det_frame.timestamp_ns;
                sync.num_detections = det_frame.num_detections;
                sync.jpeg_size    = (uint32_t)jpeg_size;
                sync.padding      = 0;
                memset(sync.detections, 0, sizeof(sync.detections));
                for (uint32_t di = 0; di < det_frame.num_detections; di++) {
                    sync.detections[di].class_id   = det_frame.detections[di].class_id;
                    sync.detections[di].confidence = det_frame.detections[di].confidence;
                    sync.detections[di].x          = det_frame.detections[di].x;
                    sync.detections[di].y          = det_frame.detections[di].y;
                    sync.detections[di].width      = det_frame.detections[di].width;
                    sync.detections[di].height     = det_frame.detections[di].height;
                }
                memcpy(sync.jpeg_data, jpeg_ptr, jpeg_size);
                syncFramePut(&sync);
            }
        }

        // ---- Step 5-7: Display pipeline (every 2nd frame) ----
        static std::vector<uint8_t> last_jpg;
        if (frame % 2 == 0 && yuv_ptr) {
            // YUV is compacted to content size (no stride) in mx_wrapper
            int disp_w = tim.content_w > 0 ? tim.content_w : src_w;
            int disp_h = content_h > 0 ? content_h : src_h;
            cv::Mat ymat(disp_h * 3 / 2, disp_w, CV_8UC1, yuv_ptr);
            cv::Mat bgr;
            cv::cvtColor(ymat, bgr, cv::COLOR_YUV2BGR_NV12);

            // Map bbox: x = x_model * sx,  y = (y_model - pad_top) * sy
            for (int i = 0; i < n; i++) {
                int off = 1 + i * 6;
                int x1 = (int)(res[off+1] * sx);
                int y1 = (int)((res[off+2] - pad_top) * sy);
                int x2 = (int)(res[off+3] * sx);
                int y2 = (int)((res[off+4] - pad_top) * sy);
                if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
                if (x2 >= disp_w) x2 = disp_w - 1;
                if (y2 >= disp_h) y2 = disp_h - 1;
                if (x2 <= x1 || y2 <= y1) continue;

                cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
                char lb[64];
                snprintf(lb, sizeof(lb), "c%d %.2f", (int)res[off], res[off+5]);
                cv::putText(bgr, lb, cv::Point(x1, std::max(15, y1 - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            }

            cv::imencode(".jpg", bgr, last_jpg, {cv::IMWRITE_JPEG_QUALITY, 60});
        }
        if (!last_jpg.empty()) {
            mjpeg.update(last_jpg);
            // Also push to POSIX shm for free_ws IPC consumers
            aclShmPutImage(last_jpg.data(), (uint32_t)last_jpg.size());
        }
        auto t3 = high_resolution_clock::now();

        // Timing
        total_cap   += duration_cast<microseconds>(t1 - t0).count();
        total_inf   += duration_cast<microseconds>(t2 - t1).count();
        total_draw  += duration_cast<microseconds>(t3 - t2).count();
        total_decode+= tim.decode_us;
        total_resize+= tim.resize_us;
        total_npu   += tim.infer_us;
        total_topk  += tim.topk_us;

        if (++frame % 30 == 0) {
            float N = 30.0f;
            fprintf(stderr, "\r\033[K[YOLO26] FPS:%.1f | Dec:%.1f Res:%.1f NPU:%.1f TopK:%.1f = Inf:%.1fms | Draw:%.1fms | Dets:%d",
                    1000.0f / ((total_cap + total_inf) / N / 1000.0f),
                    total_decode/N/1000, total_resize/N/1000, total_npu/N/1000, total_topk/N/1000,
                    total_inf/N/1000, total_draw/N/1000, n);
            fflush(stderr);
            total_cap = total_inf = total_draw = 0;
            total_decode = total_resize = total_npu = total_topk = 0;
        }
    }

    mjpeg.stop();
    if (cam_ok) cap.release();
    syncFrameWriterCleanup();
    aclShmWriterCleanup();
    mx_cleanup(ctx);
    system("v4l2-ctl -d /dev/video0 -c exposure_auto=3 2>/dev/null &");
    fprintf(stderr, "\nDone\n");
    return 0;
}
