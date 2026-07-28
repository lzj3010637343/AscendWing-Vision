#pragma once
/**
 * sot_tracker.h — C++ 8-state Kalman SOT for AscendWing unified tracker.
 *
 * State machine: SEARCHING → LOCKING → TRACKING → MEMORY
 * Uses OpenCV cv::KalmanFilter for 8-state CV model.
 * Outputs tracked bbox to /sot_control_shm for downstream consumers.
 */
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <deque>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---- State enum ----
enum SOTState { ST_SEARCHING = 0, ST_LOCKING = 1, ST_TRACKING = 2, ST_MEMORY = 3 };

// ---- Config ----
struct SOTConfig {
    float min_conf      = 0.20f;
    float lock_score    = 0.40f;
    int   locking_frames = 3;
    int   max_age        = 10;
    float iou_threshold  = 0.25f;
    float center_radius  = 80.0f;

    float conf_w   = 0.40f;
    float center_w = 0.30f;
    float size_w   = 0.20f;
    float class_w  = 0.10f;
    float center_sigma = 60.0f;
    float size_sigma   = 0.50f;

    float kf_q = 0.008f;
    float kf_r = 0.15f;
    float vel_scale = 8.0f;
    float damp_alpha = 0.90f;

    float   mem_ttl       = 1.5f;
    int     traj_frames   = 15;
    float   traj_radius   = 60.0f;
    bool    oru_enabled   = true;
    int     oru_window    = 5;
    int     target_class  = -1;
    float   dt            = 1.0f / 30.0f;
};

// ---- Control output struct (packed, matches Python) ----
#pragma pack(push, 1)
struct CtrlOut {
    uint32_t seqnum;
    uint32_t frame_id;
    float cx, cy, w, h, conf;
    int32_t track_id, state;
    uint8_t has_target;
    uint8_t _pad[3];
};
#pragma pack(pop)

// ---- Kalman wrapper ----
struct SOTKalman {
    cv::KalmanFilter kf;
    float dt;
    bool init = false;
    int age = 0;

    SOTKalman(float dt_, float q, float r) : dt(dt_) {
        kf.init(8, 4, 0, CV_32F);
        // State transition F [8x8]
        cv::setIdentity(kf.transitionMatrix);
        kf.transitionMatrix.at<float>(0,4) = dt;
        kf.transitionMatrix.at<float>(1,5) = dt;
        kf.transitionMatrix.at<float>(2,6) = dt;
        kf.transitionMatrix.at<float>(3,7) = dt;

        // Measurement H [4x8]
        kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
        kf.measurementMatrix.at<float>(0,0) = 1;
        kf.measurementMatrix.at<float>(1,1) = 1;
        kf.measurementMatrix.at<float>(2,2) = 1;
        kf.measurementMatrix.at<float>(3,3) = 1;

        // Covariances
        cv::setIdentity(kf.processNoiseCov, cv::Scalar(q));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(r));
        cv::setIdentity(kf.errorCovPost, cv::Scalar(50));

        kf.statePost = cv::Mat::zeros(8, 1, CV_32F);
    }

    void set_state(float cx, float cy, float w, float h) {
        kf.statePost.setTo(0);
        kf.statePost.at<float>(0) = cx;
        kf.statePost.at<float>(1) = cy;
        kf.statePost.at<float>(2) = w;
        kf.statePost.at<float>(3) = h;
        cv::setIdentity(kf.errorCovPost, cv::Scalar(50));
        init = true; age = 0;
    }

    cv::Mat predict() {
        cv::Mat pred = kf.predict();
        age++;
        return pred;
    }

    cv::Mat update(float cx, float cy, float w, float h) {
        cv::Mat meas(4, 1, CV_32F);
        meas.at<float>(0) = cx; meas.at<float>(1) = cy;
        meas.at<float>(2) = w;  meas.at<float>(3) = h;
        cv::Mat corr = kf.correct(meas);
        age = 0;
        return corr;
    }

    void damp(float a) {
        kf.statePost.at<float>(4) *= a;
        kf.statePost.at<float>(5) *= a;
        kf.statePost.at<float>(6) *= a;
        kf.statePost.at<float>(7) *= a;
    }

    void set_adaptive_q(float vx, float vy, float base_q, float scale) {
        float speed = std::sqrt(vx*vx + vy*vy);
        float factor = speed > 30.0f ? 1.0f + (speed - 30.0f) / 30.0f * scale : 1.0f;
        cv::setIdentity(kf.processNoiseCov, cv::Scalar(base_q * factor));
    }

    std::vector<cv::Vec4f> trajectory(int n) {
        cv::Mat x_save = kf.statePost.clone();
        cv::Mat P_save = kf.errorCovPost.clone();
        cv::Mat Q_save = kf.processNoiseCov.clone();
        std::vector<cv::Vec4f> traj;
        for (int i = 0; i < n; i++) {
            predict();
            traj.push_back(cv::Vec4f(
                kf.statePost.at<float>(0), kf.statePost.at<float>(1),
                kf.statePost.at<float>(2), kf.statePost.at<float>(3)));
        }
        kf.statePost = x_save;
        kf.errorCovPost = P_save;
        kf.processNoiseCov = Q_save;
        return traj;
    }
};

// ---- Main SOT tracker ----
struct SOTTracker {
    SOTConfig cfg;
    SOTKalman kf;
    SOTState  state = ST_SEARCHING;
    int lock_hits = 0, lost_frames = 0, track_id = 0, next_id = 1;
    double mem_enter = 0;
    std::deque<cv::Vec4f> obs_history;  // (cx,cy,w,h)
    float last_conf = 0;
    int target_class = -1;

    SOTTracker(const SOTConfig& c) : cfg(c), kf(c.dt, c.kf_q, c.kf_r) {}

    struct Result { bool has; float cx, cy, w, h, conf; int state, id; };

    // ---- Scoring ----
    float score_det(float dcx, float dcy, float dw, float dh, float dconf, int dcls,
                    float pcx, float pcy, float pw, float ph) {
        float s = cfg.conf_w * dconf;
        float dx = dcx - pcx, dy = dcy - pcy;
        float dist = std::sqrt(dx*dx + dy*dy);
        s += cfg.center_w * std::exp(-0.5f * (dist / cfg.center_sigma) * (dist / cfg.center_sigma));
        if (pw > 0 && ph > 0 && dw > 0 && dh > 0) {
            float ar = (dw * dh) / (pw * ph);
            if (ar > 0) {
                float lr = std::log(ar);
                s += cfg.size_w * std::exp(-0.5f * (lr / cfg.size_sigma) * (lr / cfg.size_sigma));
            }
        }
        if (target_class >= 0 && dcls == target_class) s += cfg.class_w;
        return s;
    }

    static float iou(float x1a, float y1a, float x2a, float y2a, float x1b, float y1b, float x2b, float y2b) {
        float ix1 = std::max(x1a, x1b), iy1 = std::max(y1a, y1b);
        float ix2 = std::min(x2a, x2b), iy2 = std::min(y2a, y2b);
        if (ix2 <= ix1 || iy2 <= iy1) return 0;
        float inter = (ix2 - ix1) * (iy2 - iy1);
        float area_a = (x2a - x1a) * (y2a - y1a);
        float area_b = (x2b - x1b) * (y2b - y1b);
        return inter / (area_a + area_b - inter);
    }

    // ---- Main update ----
    Result update(const std::vector<cv::Vec6f>& dets, int frame_id, double ts) {
        // dets: (cls, conf, x1, y1, x2, y2)  — x1,y1,x2,y2 in display coords
        if (state == ST_SEARCHING) return handle_searching(dets, ts);
        if (state == ST_LOCKING)   return handle_locking(dets, ts);
        if (state == ST_TRACKING)  return handle_tracking(dets, ts);
        if (state == ST_MEMORY)    return handle_memory(dets, ts);
        return {false, 0,0,0,0, 0, ST_SEARCHING, 0};
    }

    Result handle_searching(const std::vector<cv::Vec6f>& dets, double ts) {
        float best_s = cfg.lock_score;
        int best_i = -1;
        for (size_t i = 0; i < dets.size(); i++) {
            float dcx = (dets[i][2] + dets[i][4]) * 0.5f;
            float dcy = (dets[i][3] + dets[i][5]) * 0.5f;
            float s = cfg.conf_w * dets[i][1];
            if (cfg.target_class < 0 || (int)dets[i][0] == cfg.target_class) {
                if (s > best_s) { best_s = s; best_i = (int)i; }
            }
        }
        if (best_i >= 0) {
            float dcx = (dets[best_i][2] + dets[best_i][4]) * 0.5f;
            float dcy = (dets[best_i][3] + dets[best_i][5]) * 0.5f;
            float dw = dets[best_i][4] - dets[best_i][2];
            float dh = dets[best_i][5] - dets[best_i][3];
            kf.set_state(dcx, dcy, dw, dh);
            lock_hits = 1; lost_frames = 0;
            last_conf = dets[best_i][1];
            target_class = (int)dets[best_i][0];
            obs_history.clear();
            obs_history.push_back(cv::Vec4f(dcx, dcy, dw, dh));
            state = ST_LOCKING;
        }
        return {false, 0,0,0,0, 0, ST_SEARCHING, 0};
    }

    Result handle_locking(const std::vector<cv::Vec6f>& dets, double ts) {
        cv::Mat pred = kf.predict();
        float pcx = pred.at<float>(0), pcy = pred.at<float>(1);
        float pw = pred.at<float>(2), ph = pred.at<float>(3);

        float best_s = 0; int best_i = -1;
        for (size_t i = 0; i < dets.size(); i++) {
            float dcx = (dets[i][2] + dets[i][4]) * 0.5f;
            float dcy = (dets[i][3] + dets[i][5]) * 0.5f;
            float dw = dets[i][4] - dets[i][2], dh = dets[i][5] - dets[i][3];
            float s = score_det(dcx, dcy, dw, dh, dets[i][1], (int)dets[i][0], pcx, pcy, pw, ph);
            if (s > best_s) { best_s = s; best_i = (int)i; }
        }

        if (best_i >= 0 && best_s >= cfg.lock_score) {
            float dcx = (dets[best_i][2] + dets[best_i][4]) * 0.5f;
            float dcy = (dets[best_i][3] + dets[best_i][5]) * 0.5f;
            float dw = dets[best_i][4] - dets[best_i][2];
            float dh = dets[best_i][5] - dets[best_i][3];
            kf.update(dcx, dcy, dw, dh);
            lock_hits++; last_conf = dets[best_i][1];
            obs_history.push_back(cv::Vec4f(dcx, dcy, dw, dh));
            if ((int)obs_history.size() > cfg.oru_window) obs_history.pop_front();

            if (lock_hits >= cfg.locking_frames) {
                state = ST_TRACKING; lost_frames = 0;
                track_id = next_id++;
            }
            if (state == ST_TRACKING)
                return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                        kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                        last_conf, ST_TRACKING, track_id};
        } else {
            state = ST_SEARCHING; lock_hits = 0; obs_history.clear();
        }
        return {false, 0,0,0,0, 0, ST_SEARCHING, 0};
    }

    Result handle_tracking(const std::vector<cv::Vec6f>& dets, double ts) {
        cv::Mat pred = kf.predict();
        float pcx = pred.at<float>(0), pcy = pred.at<float>(1);
        float pw = pred.at<float>(2), ph = pred.at<float>(3);

        // Adaptive Q
        float vx = kf.kf.statePost.at<float>(4), vy = kf.kf.statePost.at<float>(5);
        kf.set_adaptive_q(vx, vy, cfg.kf_q, cfg.vel_scale);

        // Primary: 4D score
        float best_s = 0; int best_i = -1;
        for (size_t i = 0; i < dets.size(); i++) {
            float dcx = (dets[i][2] + dets[i][4]) * 0.5f;
            float dcy = (dets[i][3] + dets[i][5]) * 0.5f;
            float dw = dets[i][4] - dets[i][2], dh = dets[i][5] - dets[i][3];
            float s = score_det(dcx, dcy, dw, dh, dets[i][1], (int)dets[i][0], pcx, pcy, pw, ph);
            if (s > best_s) { best_s = s; best_i = (int)i; }
        }

        if (best_i >= 0 && best_s >= cfg.lock_score * 0.8f) {
            float dcx = (dets[best_i][2] + dets[best_i][4]) * 0.5f;
            float dcy = (dets[best_i][3] + dets[best_i][5]) * 0.5f;
            float dw = dets[best_i][4] - dets[best_i][2];
            float dh = dets[best_i][5] - dets[best_i][3];
            kf.update(dcx, dcy, dw, dh);
            last_conf = dets[best_i][1];
            obs_history.push_back(cv::Vec4f(dcx, dcy, dw, dh));
            if ((int)obs_history.size() > cfg.oru_window) obs_history.pop_front();
            lost_frames = 0;
            return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                    kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                    last_conf, ST_TRACKING, track_id};
        }

        // IoU fallback
        float best_iou = cfg.iou_threshold; int bi = -1;
        float p_x1 = pcx - pw * 0.5f, p_y1 = pcy - ph * 0.5f;
        float p_x2 = pcx + pw * 0.5f, p_y2 = pcy + ph * 0.5f;
        for (size_t i = 0; i < dets.size(); i++) {
            float ii = iou(dets[i][2], dets[i][3], dets[i][4], dets[i][5], p_x1, p_y1, p_x2, p_y2);
            if (ii >= best_iou) {
                float dcx = (dets[i][2] + dets[i][4]) * 0.5f;
                float dcy = (dets[i][3] + dets[i][5]) * 0.5f;
                float dist = std::sqrt((dcx-pcx)*(dcx-pcx) + (dcy-pcy)*(dcy-pcy));
                if (dist < cfg.center_radius) { best_iou = ii; bi = (int)i; }
            }
        }
        if (bi >= 0) {
            float dcx = (dets[bi][2] + dets[bi][4]) * 0.5f;
            float dcy = (dets[bi][3] + dets[bi][5]) * 0.5f;
            float dw = dets[bi][4] - dets[bi][2], dh = dets[bi][5] - dets[bi][3];
            kf.update(dcx, dcy, dw, dh);
            last_conf = dets[bi][1];
            obs_history.push_back(cv::Vec4f(dcx, dcy, dw, dh));
            if ((int)obs_history.size() > cfg.oru_window) obs_history.pop_front();
            lost_frames = 0;
            return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                    kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                    last_conf, ST_TRACKING, track_id};
        }

        lost_frames++;
        if (lost_frames >= cfg.max_age) {
            state = ST_MEMORY; mem_enter = ts;
        }
        return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                last_conf, state == ST_MEMORY ? ST_MEMORY : ST_TRACKING, track_id};
    }

    Result handle_memory(const std::vector<cv::Vec6f>& dets, double ts) {
        if (ts - mem_enter > cfg.mem_ttl) {
            state = ST_SEARCHING; obs_history.clear();
            return {false, 0,0,0,0, 0, ST_SEARCHING, 0};
        }

        kf.damp(cfg.damp_alpha);
        auto traj = kf.trajectory(cfg.traj_frames);

        float best_iou = cfg.iou_threshold; int bi = -1;
        for (int step = 0; step < (int)traj.size(); step++) {
            float tcx = traj[step][0], tcy = traj[step][1];
            float tw = traj[step][2], th = traj[step][3];
            for (size_t i = 0; i < dets.size(); i++) {
                float ii = iou(dets[i][2], dets[i][3], dets[i][4], dets[i][5],
                               tcx - tw*0.5f, tcy - th*0.5f, tcx + tw*0.5f, tcy + th*0.5f);
                if (ii > best_iou) {
                    float dcx = (dets[i][2] + dets[i][4]) * 0.5f;
                    float dcy = (dets[i][3] + dets[i][5]) * 0.5f;
                    float dist = std::sqrt((dcx-tcx)*(dcx-tcx) + (dcy-tcy)*(dcy-tcy));
                    if (dist < cfg.traj_radius * (step + 1)) {
                        best_iou = ii; bi = (int)i;
                    }
                }
            }
        }

        if (bi >= 0) {
            // Re-acquired
            float dcx = (dets[bi][2] + dets[bi][4]) * 0.5f;
            float dcy = (dets[bi][3] + dets[bi][5]) * 0.5f;
            float dw = dets[bi][4] - dets[bi][2], dh = dets[bi][5] - dets[bi][3];

            // ORU: replay observation history
            if (cfg.oru_enabled && !obs_history.empty()) {
                for (auto& o : obs_history) {
                    kf.update(o[0], o[1], o[2], o[3]);
                }
            }
            kf.update(dcx, dcy, dw, dh);
            last_conf = dets[bi][1];
            obs_history.push_back(cv::Vec4f(dcx, dcy, dw, dh));
            if ((int)obs_history.size() > cfg.oru_window) obs_history.pop_front();
            state = ST_TRACKING; lost_frames = 0;
            return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                    kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                    last_conf, ST_TRACKING, track_id};
        }

        // Still lost — predict forward
        kf.predict();
        return {true, kf.kf.statePost.at<float>(0), kf.kf.statePost.at<float>(1),
                kf.kf.statePost.at<float>(2), kf.kf.statePost.at<float>(3),
                last_conf, ST_MEMORY, track_id};
    }

    void reset() {
        state = ST_SEARCHING; lock_hits = 0; lost_frames = 0;
        obs_history.clear(); target_class = -1;
        kf = SOTKalman(cfg.dt, cfg.kf_q, cfg.kf_r);
    }
};

// ---- Control shm writer (mmap, no semaphore) ----
struct CtrlWriter {
    int fd = -1; void* ptr = nullptr; uint32_t seq = 0;
    bool ok = false;

    bool init(const char* path = "/sot_control_shm") {
        std::string f = "/dev/shm" + std::string(path);
        fd = open(f.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd < 0) { perror("[SOT] ctrl open"); return false; }
        if (ftruncate(fd, sizeof(CtrlOut)) != 0) { close(fd); return false; }
        ptr = mmap(nullptr, sizeof(CtrlOut), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); return false; }
        close(fd); fd = -1;
        ok = true;
        fprintf(stderr, "[SOT] Control shm ready (%zu bytes)\n", sizeof(CtrlOut));
        return true;
    }

    void write(int frame_id, float cx, float cy, float w, float h, float conf,
               int state, int tid, bool has_target) {
        if (!ok) return;
        CtrlOut c;
        c.seqnum = ++seq; c.frame_id = (uint32_t)frame_id;
        c.cx = cx; c.cy = cy; c.w = w; c.h = h; c.conf = conf;
        c.track_id = tid; c.state = state;
        c.has_target = has_target ? 1 : 0;
        memset(c._pad, 0, 3);
        memcpy(ptr, &c, sizeof(c));
        msync(ptr, sizeof(c), MS_SYNC);
    }

    void cleanup() {
        if (ptr && ptr != MAP_FAILED) munmap(ptr, sizeof(CtrlOut));
        if (fd >= 0) close(fd);
        shm_unlink("/sot_control_shm");
    }
};
