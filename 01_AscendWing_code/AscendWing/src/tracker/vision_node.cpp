/**
 * vision_node.cpp — YOLO vision node (yolov5cpp pipeline → /sync_frame_shm)
 *
 * Replaces unified tracker's YOLO pipeline with yolov5cpp-style:
 *   V4L2 MJPEG → MxBase decode (YUV420SP) → direct NPU inference
 *   → parse YOLO raw output → NMS → write to /sync_frame_shm
 *
 * No CropResize, no AIPP padding — works with any YOLOv5 .om model.
 *
 * Downstream (unchanged):
 *   sot_live.py     — reads /sync_frame_shm → SOT + MJPEG display + /sot_control_shm
 *
 * Build:  make vision_node
 * Usage:  ./run.sh vision [model.om] [--conf 0.35] [--offx 160 --offy 40]
 */
#include "sync_frame_shm.h"

#include <MxBase/MxBase.h>
#include <MxBase/E2eInfer/Tensor/Tensor.h>
#include <MxBase/E2eInfer/Image/Image.h>
#include <MxBase/E2eInfer/Model/Model.h>
#include <MxBase/E2eInfer/ImageProcessor/ImageProcessor.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <chrono>
#include <vector>
#include <fstream>

using namespace MxBase;
using namespace std::chrono;

// ---- Configurable ----
static int   IMG_W       = 960;
static int   IMG_H       = 720;
static float CONF_THRESH = 0.35f;
static float NMS_THRESH  = 0.4f;
static int   AIPP_CX_OFF = 160;  // YOLOv5 ball model: (960-640)/2
static int   AIPP_CY_OFF = 40;   // YOLOv5 ball model: (720-640)/2
static const char* DEFAULT_MODEL = "models/ball_aipp.om";

static volatile bool g_run = true;
static void sig_handler(int) { g_run = false; }

int main(int argc, char** argv) {
    const char* model_path = DEFAULT_MODEL;
    if (argc > 1 && argv[1][0] != '-') model_path = argv[1];

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--conf")  && i + 1 < argc) CONF_THRESH = atof(argv[++i]);
        else if (!strcmp(argv[i], "--nms")   && i + 1 < argc) NMS_THRESH  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--offx")  && i + 1 < argc) AIPP_CX_OFF = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--offy")  && i + 1 < argc) AIPP_CY_OFF = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) IMG_W       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height")&& i + 1 < argc) IMG_H       = atoi(argv[++i]);
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // ---- Init mxVision ----
    fprintf(stderr, "[VISION] Model: %s\n", model_path);
    fprintf(stderr, "[VISION] YOLO pipeline: yolov5cpp-style (YUV420SP → NPU, no CropResize)\n");
    if (MxInit() != APP_ERR_OK) { fprintf(stderr, "[VISION] MxInit failed\n"); return 1; }

    std::string model_str(model_path);
    Model model(model_str, 0);
    ImageProcessor imageProcessor(0);
    { auto ish=model.GetInputTensorShape(); auto ifmt=model.GetInputFormat(); auto idt=model.GetInputTensorDataType();
      fprintf(stderr,"[VISION] model input: fmt=%d dtype=%d shape=[",(int)ifmt,(int)idt);
      for(auto d:ish) fprintf(stderr,"%lld,",(long long)d);
      fprintf(stderr,"]\n"); }

    // ---- Init sync frame shm (→ sot_live.py) ----
    if (syncFrameWriterInit() != SYNC_OK) {
        fprintf(stderr, "[VISION] sync shm init failed\n");
        return 1;
    }
    fprintf(stderr, "[VISION] /sync_frame_shm ready (%zu bytes)\n", sizeof(AShmSyncFrame));

    // ---- Camera ----
    cv::VideoCapture cap;
    bool cam_ok = false;
    for (int i = 0; i <= 3; i++) {
        cap.open(i, cv::CAP_V4L2);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  IMG_W);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);  // model expects 1080H
            cap.set(cv::CAP_PROP_FPS, 90);
            cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
            cv::Mat t;
            if (cap.read(t) && !t.empty()) { cam_ok = true; break; }
            cap.release();
        }
    }

    std::vector<uint8_t> file_jpg;
    if (!cam_ok) {
        // Try fallback JPEG for headless testing
        std::ifstream ifs("/root/yolo26/test.jpg", std::ios::binary);
        if (ifs) {
            file_jpg.assign(std::istreambuf_iterator<char>(ifs),
                           std::istreambuf_iterator<char>());
            fprintf(stderr, "[VISION] No camera, using file: %zu bytes\n", file_jpg.size());
        } else {
            fprintf(stderr, "[VISION] No camera, no test image. Exiting.\n");
            return 1;
        }
    } else {
        fprintf(stderr, "[VISION] Camera %dx%d MJPEG ready\n", IMG_W, IMG_H);
    }

    // ---- Main loop ----
    int frame = 0;
    long total_ms = 0;

    fprintf(stderr, "[VISION] Running (→ /sync_frame_shm → sot_live.py → downstream)...\n");

    while (g_run) {
        auto t0 = high_resolution_clock::now();

        // Capture
        std::vector<uint8_t> cam_buf;
        uint8_t* jpeg_ptr;
        int      jpeg_size;
        if (cam_ok) {
            cv::Mat mjpeg;
            if (!cap.read(mjpeg) || mjpeg.empty()) continue;
            cam_buf.assign(mjpeg.data, mjpeg.data + mjpeg.total() * mjpeg.elemSize());
            jpeg_ptr  = cam_buf.data();
            jpeg_size = (int)cam_buf.size();
        } else {
            jpeg_ptr  = file_jpg.data();
            jpeg_size = (int)file_jpg.size();
        }

        // ---- OpenCV: decode JPEG → BGR → resize to 960x720 for AIPP ----
        // OpenCV decode -> resize 960x1080 (model AIPP expects 1080H) -> NV12
// Decode JPEG -> YUV420SP 960x720 (camera native, matches model input slot)
        Image img;
        std::shared_ptr<uint8_t> jpgShm(jpeg_ptr, [](uint8_t*){});
        if (imageProcessor.Decode(jpgShm, jpeg_size, img, ImageFormat::YUV_SP_420) != APP_ERR_OK) continue;
        if (frame < 3) { auto _s = img.GetSize(); fprintf(stderr, "[DBG] decoded img=%dx%d\n", (int)_s.width, (int)_s.height); }

        // ---- NPU Inference ----
        auto t1 = high_resolution_clock::now();
        std::vector<Tensor> inputs = { img.ConvertToTensor() };
        if (frame < 3) { auto _ts = inputs[0].GetShape(); fprintf(stderr, "[DBG] tensor shape=["); for(auto _d:_ts) fprintf(stderr, "%lld,", (long long)_d); fprintf(stderr, "]\n"); }
        auto outputs = model.Infer(inputs);
        outputs[0].ToHost();

        float* out  = (float*)outputs[0].GetData();
        auto  shape = outputs[0].GetShape();
        int   nrow  = (int)shape[1];
        int   ncol  = (int)shape[2];

        // [DEBUG] diagnose Dets:0 - output shape + obj distribution
        if (frame < 5) {
            char fn[64]; snprintf(fn,sizeof(fn),"/tmp/vis_frame%d.jpg",frame);
            FILE* fp=fopen(fn,"wb"); if(fp){fwrite(jpeg_ptr,1,jpeg_size,fp); fclose(fp);}
            fprintf(stderr," [jpg=%d bytes]",jpeg_size);
            float max_obj = -1.f; int max_i = -1;
            for (int i = 0; i < nrow; i++) {
                float o = out[i * ncol + 4];
                if (o > max_obj) { max_obj = o; max_i = i; }
            }
            fprintf(stderr, "\n[DBG] frame=%d shape=[%d,%d] max_obj=%.4f@row%d", frame, nrow, ncol, max_obj, max_i);
            if (max_i >= 0) {
                fprintf(stderr, " cx=%.1f cy=%.1f w=%.1f h=%.1f obj=%.3f",
                        out[max_i*ncol+0], out[max_i*ncol+1], out[max_i*ncol+2], out[max_i*ncol+3], out[max_i*ncol+4]);
            }
            fprintf(stderr, "\n");
        }

        auto t2 = high_resolution_clock::now();

        // Parse YOLO output → detections
        struct Det { float cx, cy, w, h, conf; int cls; };
        std::vector<Det> raw_dets;

        for (int i = 0; i < nrow; i++) {
            float obj = out[i * ncol + 4];
            if (obj < CONF_THRESH) continue;

            int   best_cls = 0;
            float best_sc  = 0;
            if (ncol > 5) {
                for (int c = 5; c < ncol; c++) {
                    if (out[i * ncol + c] > best_sc) {
                        best_sc = out[i * ncol + c]; best_cls = c - 5;
                    }
                }
            }
            float conf = obj * (ncol > 5 ? best_sc : 1.0f);

            Det d;
            // Same logic as yolov5cpp/main.cpp:
            // ball_aipp.om AIPP: crop 960x720 → 640x640 + resize to 640x640.
            // Model outputs cx/cy/w/h in 640 input space.
            // yolov5cpp adds 160/40 to map back to 960x720 display.
            // It does NOT scale w/h — they work at the crop's 640 space.
            d.cx   = out[i * ncol + 0] + AIPP_CX_OFF;
            d.cy   = out[i * ncol + 1] + AIPP_CY_OFF;
            d.w    = out[i * ncol + 2];
            d.h    = out[i * ncol + 3];
            d.conf = conf;
            d.cls  = best_cls;
            raw_dets.push_back(d);
        }

        // NMS
        std::vector<int> idx;
        {
            std::vector<cv::Rect> boxes;
            std::vector<float>   scores;
            for (auto& d : raw_dets) {
                boxes.push_back(cv::Rect(d.cx - d.w/2, d.cy - d.h/2, d.w, d.h));
                scores.push_back(d.conf);
            }
            if (!boxes.empty())
                cv::dnn::NMSBoxes(boxes, scores, NMS_THRESH, NMS_THRESH, idx);
        }

        // ---- Write sync frame ----
        AShmSyncFrame sync;
        memset(&sync, 0, sizeof(sync));
        sync.frame_id      = (uint32_t)frame;
        sync.timestamp_ns  = (uint64_t)duration_cast<nanoseconds>(
            high_resolution_clock::now().time_since_epoch()).count();

        int n_det = 0;
        for (int i : idx) {
            if (n_det >= ASHM_MAX_DETECTIONS) break;
            auto& d = raw_dets[i];
            auto& sd = sync.detections[n_det];
            sd.class_id   = d.cls;
            sd.confidence = d.conf;
            sd.x      = d.cx - d.w / 2;
            sd.y      = d.cy - d.h / 2;
            sd.width  = d.w;
            sd.height = d.h;
            n_det++;
        }
        sync.num_detections = (uint32_t)n_det;
        sync.jpeg_size      = (uint32_t)jpeg_size;
        memcpy(sync.jpeg_data, jpeg_ptr, jpeg_size);
        syncFramePut(&sync);

        // Stats
        auto t3 = high_resolution_clock::now();
        float ms = duration_cast<microseconds>(t3 - t0).count() / 1000.0f;
        total_ms += (long)(ms * 1000);
        frame++;

        if (frame % 30 == 0) {
            // Debug: print first detection coords (check AIPP offset)
            if (n_det > 0) {
                auto& d0 = sync.detections[0];
                fprintf(stderr, "\n[VISION] det: xy=(%.0f,%.0f) wh=(%.0f,%.0f) conf=%.2f cls=%d\n",
                        d0.x, d0.y, d0.width, d0.height, d0.confidence, d0.class_id);
            }
            float inf_ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0f;
            fprintf(stderr, "[VISION] FPS:%5.1f | Dets:%d | Inf:%.1fms Total:%.1fms\r",
                    1000.0f / (ms), n_det, inf_ms, ms);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n[VISION] Stopped. %d frames\n", frame);

    // Cleanup
    syncFrameWriterCleanup();
    if (cam_ok) cap.release();
    MxDeInit();
    return 0;
}
