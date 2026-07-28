#pragma once
// Simple EIS: sparse LK on Y plane → median translation → VPC crop offset
// Operates on NV12 Y plane, ~2-3ms @ 800x600 on ARM A55
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

struct EisState {
    cv::Mat prev_y;                    // previous frame Y plane
    std::vector<cv::Point2f> prev_pts; // tracked feature points
    float cum_dx = 0, cum_dy = 0;     // cumulative smoothed translation
    int frame = 0;
    bool initialized = false;

    // Tuning knobs
    int max_pts = 200;                 // FAST corner count
    float smooth_alpha = 0.7f;        // low-pass filter (higher = smoother, slower response)
    float dead_zone = 0.5f;           // ignore motion < 0.5px
    float max_correction = 15.0f;     // max per-frame correction (avoid jump on scene change)
    int refine_every = 3;             // re-detect features every N frames

    // Output: crop offset for VPC (positive = content moved → shift crop opposite)
    int crop_dx = 0, crop_dy = 0;
};

// Initialize with first frame's Y plane
inline void eis_init(EisState& s, const cv::Mat& y_plane) {
    s.prev_y = y_plane.clone();
    s.cum_dx = s.cum_dy = 0;
    s.crop_dx = s.crop_dy = 0;
    s.frame = 0;
    s.initialized = false;
}

// Process one frame: returns true if stabilization was applied
inline bool eis_process(EisState& s, const cv::Mat& y_plane) {
    if (y_plane.empty()) return false;
    int w = y_plane.cols, h = y_plane.rows;
    s.frame++;

    // ---- Re-detect features periodically ----
    if (s.frame == 1 || s.frame % s.refine_every == 0 || s.prev_pts.size() < 50) {
        std::vector<cv::Point2f> corners;
        // Grid-based FAST: divide image into 8×6 cells, pick strongest in each
        int cx = 8, cy = 6;
        int cw = w / cx, ch = h / cy;
        for (int gy = 0; gy < cy; gy++) {
            for (int gx = 0; gx < cx; gx++) {
                cv::Rect roi(gx * cw + cw/4, gy * ch + ch/4, cw/2, ch/2);
                if (roi.x + roi.width > w) roi.width = w - roi.x;
                if (roi.y + roi.height > h) roi.height = h - roi.y;
                if (roi.width < 20 || roi.height < 20) continue;
                std::vector<cv::Point2f> cell_corners;
                cv::goodFeaturesToTrack(y_plane(roi), cell_corners, 5, 0.05, 5);
                for (auto& p : cell_corners) {
                    p.x += roi.x; p.y += roi.y;
                    corners.push_back(p);
                }
            }
        }
        // Limit total points
        if ((int)corners.size() > s.max_pts) {
            std::nth_element(corners.begin(), corners.begin() + s.max_pts, corners.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.x*a.x + a.y*a.y > b.x*b.x + b.y*b.y; });
            corners.resize(s.max_pts);
        }
        s.prev_pts = corners;
        s.prev_y = y_plane.clone();
        if (!s.initialized) s.initialized = true;
        return false;  // first frame or re-detect: no motion estimate
    }

    if (s.prev_pts.empty()) return false;

    // ---- Lucas-Kanade optical flow ----
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(s.prev_y, y_plane, s.prev_pts, curr_pts, status, err,
                              cv::Size(15, 15), 3);  // 15×15 window, 3-level pyramid

    // ---- Median translation ----
    std::vector<float> dxs, dys;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            dxs.push_back(curr_pts[i].x - s.prev_pts[i].x);
            dys.push_back(curr_pts[i].y - s.prev_pts[i].y);
        }
    }
    if (dxs.size() < 10) return false;

    size_t mid = dxs.size() / 2;
    std::nth_element(dxs.begin(), dxs.begin() + mid, dxs.end());
    std::nth_element(dys.begin(), dys.begin() + mid, dys.end());
    float dx = dxs[mid], dy = dys[mid];

    // ---- Dead zone ----
    if (fabs(dx) < s.dead_zone) dx = 0;
    if (fabs(dy) < s.dead_zone) dy = 0;

    // ---- Low-pass filter (smooth motion) ----
    s.cum_dx = s.smooth_alpha * s.cum_dx + (1.0f - s.smooth_alpha) * dx;
    s.cum_dy = s.smooth_alpha * s.cum_dy + (1.0f - s.smooth_alpha) * dy;

    // ---- Clamp per-frame correction ----
    float corr_x = copysign(fminf(fabsf(s.cum_dx), s.max_correction), s.cum_dx);
    float corr_y = copysign(fminf(fabsf(s.cum_dy), s.max_correction), s.cum_dy);

    // VPC crop offset: camera moved right (+dx) → shift crop left (-dx)
    s.crop_dx = (int)(-corr_x);
    s.crop_dy = (int)(-corr_y);

    // ---- Update tracking ----
    // Keep successfully tracked points for next frame
    std::vector<cv::Point2f> good_pts;
    for (size_t i = 0; i < status.size(); i++)
        if (status[i]) good_pts.push_back(curr_pts[i]);
    s.prev_pts = good_pts;
    s.prev_y = y_plane.clone();

    return true;
}
