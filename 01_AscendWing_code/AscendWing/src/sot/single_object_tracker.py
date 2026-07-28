#!/usr/bin/env python3
"""
single_object_tracker.py — Single Object Tracking (SOT) for AscendWing

State machine:
  SEARCHING → LOCKING → TRACKING → VERIFYING → SWITCHING
      ↑                      ↓           ↓           ↓
      └────── LOST ←─────────┴───────────┴───────────┘

Key features:
- Kalman filter (8-state constant velocity, mirrors catkin_ws SORT)
- Four-component scoring: confidence + center + size + class
- Wrong-target detection: confidence drop, class mismatch, aspect drift
- OC-SORT improvements: OCM (direction bonus), OCR (last-obs recovery), ORU (re-update)
- Cubic smoothstep blend for seamless target switching
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class SOTState(Enum):
    SEARCHING = 0
    LOCKING   = 1
    TRACKING  = 2
    VERIFYING = 3
    SWITCHING = 4


@dataclass
class TrackedTarget:
    cx: float; cy: float; w: float; h: float
    conf: float; cls: int; track_id: int
    state: SOTState
    is_blending: bool = False
    blend_progress: float = 0.0


@dataclass
class SOTConfig:
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1
    lock_score_threshold: float = 0.55
    lock_consistency_frames: int = 3
    lock_timeout_frames: int = 5
    min_hits: int = 3
    max_age: int = 30
    iou_threshold: float = 0.3
    iou_expand_margin: float = 1.0
    conf_floor: float = 0.35
    conf_drop_frames: int = 5
    class_mismatch_frames: int = 3
    aspect_drift_factor: float = 1.8
    verify_clear_frames: int = 3
    switch_score_threshold: float = 0.5
    blend_frames: int = 10
    ocm_enabled: bool = True
    ocm_delta_t: int = 3
    ocm_weight: float = 0.2
    ocr_enabled: bool = True
    oru_enabled: bool = True
    oru_max_gap: int = 60
    score_conf_weight: float = 0.40
    score_center_weight: float = 0.30
    score_size_weight: float = 0.20
    score_class_weight: float = 0.10
    center_ideal_x: float = 320.0
    center_ideal_y: float = 288.0
    center_sigma_x: float = 160.0
    center_sigma_y: float = 120.0
    size_ideal_area: float = 2500.0
    size_log_sigma: float = 1.5
    target_class: int = -1
    min_confidence: float = 0.25
    frame_w: float = 800.0
    frame_h: float = 600.0


def _iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    x1, y1 = cx1 - w1/2, cy1 - h1/2
    x2, y2 = cx1 + w1/2, cy1 + h1/2
    u1, v1 = cx2 - w2/2, cy2 - h2/2
    u2, v2 = cx2 + w2/2, cy2 + h2/2
    ix1, iy1 = max(x1, u1), max(y1, v1)
    ix2, iy2 = min(x2, u2), min(y2, v2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    uni = w1 * h1 + w2 * h2 - inter
    return inter / uni if uni > 0 else 0.0


def _iou_expand(pcx, pcy, pw, ph, dcx, dcy, dw, dh, thresh=0.3, margin=1.0):
    """Expanded IoU matching (C++ iou_match).
    If IoU under threshold, override with center-distance check.
    Higher margin = stickier tracking.
    radius = max(w,h) * (0.5 + margin).  margin=1.0 → radius=1.5×box."""
    v = _iou(pcx, pcy, pw, ph, dcx, dcy, dw, dh)
    if v >= thresh or margin <= 0:
        return v
    cd = ((dcx - pcx)**2 + (dcy - pcy)**2)**0.5
    md = max(pw, ph) * (0.5 + margin)
    return thresh + 0.001 if cd < md else v


class KalmanBoxTracker:
    def __init__(self, cx, cy, w, h, process_noise=0.01, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4): self.kf.transitionMatrix[i, i+4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4): self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)
        self.hits = 1; self.age = 0; self.time_since_update = 0

    def predict(self):
        state = self.kf.predict(); self.age += 1; self.time_since_update += 1
        return (float(state[0]), float(state[1]), float(state[2]), float(state[3]))

    def predict_only(self):
        state = self.kf.predict(); self.age += 1
        return (float(state[0]), float(state[1]), float(state[2]), float(state[3]))

    def update(self, cx, cy, w, h):
        m = np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        self.kf.correct(m); self.hits += 1; self.time_since_update = 0

    @property
    def box(self):
        s = self.kf.statePost
        return (float(s[0]), float(s[1]), float(s[2]), float(s[3]))

    def oru_reupdate(self, old_obs, new_obs, gap_frames):
        if gap_frames < 2: return
        ocx, ocy, ow, oh = old_obs
        ncx, ncy, nw, nh = new_obs
        d_cx = (ncx - ocx) / gap_frames; d_cy = (ncy - ocy) / gap_frames
        d_w = (nw - ow) / gap_frames; d_h = (nh - oh) / gap_frames
        for step in range(1, gap_frames):
            v_cx = ocx + d_cx*step; v_cy = ocy + d_cy*step
            v_w = ow + d_w*step; v_h = oh + d_h*step
            self.kf.predict()
            self.kf.correct(np.array([[v_cx],[v_cy],[v_w],[v_h]], dtype=np.float32))
            self.hits += 1; self.age += 1


def _score_confidence(conf): return conf

def _score_center(cx, cy, cfg):
    dx = (cx - cfg.center_ideal_x) / cfg.center_sigma_x
    dy = (cy - cfg.center_ideal_y) / cfg.center_sigma_y
    return math.exp(-0.5 * (dx*dx + dy*dy))

def _score_size(area, cfg):
    if area <= 0: return 0.0
    log_diff = abs(math.log(area) - math.log(cfg.size_ideal_area))
    return math.exp(-log_diff / cfg.size_log_sigma)

def _score_class(cls, cfg):
    if cfg.target_class < 0: return 1.0
    return 1.0 if cls == cfg.target_class else 0.3

def _score_detection(cls, conf, cx, cy, area, cfg):
    return (cfg.score_conf_weight * _score_confidence(conf) +
            cfg.score_center_weight * _score_center(cx, cy, cfg) +
            cfg.score_size_weight * _score_size(area, cfg) +
            cfg.score_class_weight * _score_class(cls, cfg))

def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t*t*t / (t*t*t + (1.0-t)*(1.0-t)*(1.0-t))


class SingleObjectTracker:
    def __init__(self, config=SOTConfig()):
        self.cfg = config; self.state = SOTState.SEARCHING
        self._tracker = None; self._next_id = 0
        self._locked_cls = -1; self._locked_aspect = 1.0; self._locked_box = (0,0,0,0)
        self._conf_ewma = 0.0; self._conf_ewma_alpha = 0.3
        self._lock_consistent = 0; self._lock_hits = 0; self._lost_frames = 0
        self._conf_drop_count = 0; self._class_mismatch_count = 0; self._verify_clear_count = 0
        self._switch_frame = 0; self._new_tracker = None; self._old_last_pos = (0,0,0,0)
        self._best_candidate = None
        self._obs_history = deque(maxlen=5)
        self._saved_tracker = None; self._saved_last_obs = (0,0,0,0); self._saved_loss_frame = -1

    def update(self, detections, timestamp):
        dets = []
        for d in detections:
            conf = float(d.get("conf", d.get("confidence", 0)))
            if conf < self.cfg.min_confidence: continue
            cls = int(d.get("cls", d.get("class_id", -1)))
            if self.cfg.target_class >= 0 and cls != self.cfg.target_class: continue
            x, y = float(d["x"]), float(d["y"])
            w, h = float(d["w"]), float(d["h"])
            dets.append((cls, conf, x+w/2, y+h/2, w, h))
        self._transition(dets, timestamp)
        return self._produce_output()

    def reset(self):
        self.state = SOTState.SEARCHING
        self._tracker = None; self._new_tracker = None; self._locked_cls = -1
        self._conf_ewma = 0.0; self._lock_consistent = 0; self._lock_hits = 0
        self._lost_frames = 0; self._conf_drop_count = 0; self._class_mismatch_count = 0
        self._verify_clear_count = 0; self._switch_frame = 0; self._best_candidate = None
        self._obs_history.clear()
        self._saved_tracker = None; self._saved_last_obs = (0,0,0,0); self._saved_loss_frame = -1

    @property
    def current_target(self): return self._produce_output()

    def _transition(self, dets, ts):
        if self.state == SOTState.SEARCHING: self._transition_searching(dets)
        elif self.state == SOTState.LOCKING: self._transition_locking(dets)
        elif self.state == SOTState.TRACKING: self._transition_tracking(dets)
        elif self.state == SOTState.VERIFYING: self._transition_verifying(dets)
        elif self.state == SOTState.SWITCHING: self._transition_switching(dets)

    def _transition_searching(self, dets):
        if not dets: self._lock_consistent = 0; self._best_candidate = None; return
        best, best_score = None, 0.0
        for cls, conf, cx, cy, w, h in dets:
            s = _score_detection(cls, conf, cx, cy, w*h, self.cfg)
            if s > best_score: best_score = s; best = (cls, conf, cx, cy, w, h, s)
        if best is None or best_score < self.cfg.lock_score_threshold:
            self._lock_consistent = 0; self._best_candidate = None; return
        if self._best_candidate is not None:
            prev = self._best_candidate
            iou_val = _iou(prev[2], prev[3], prev[4], prev[5], best[2], best[3], best[4], best[5])
            self._lock_consistent = self._lock_consistent + 1 if iou_val > 0.5 else 1
        else:
            self._lock_consistent = 1
        self._best_candidate = best
        if self._lock_consistent >= self.cfg.lock_consistency_frames:
            cls, conf, cx, cy, w, h, _ = best
            if (self.cfg.oru_enabled and self._saved_tracker is not None and self._saved_last_obs != (0,0,0,0)):
                slx, sly, slw, slh = self._saved_last_obs; gap = self._saved_loss_frame
                svx = self._saved_tracker.kf.statePost[4][0]; svy = self._saved_tracker.kf.statePost[5][0]
                expected_cx = slx + svx * gap; expected_cy = sly + svy * gap
                dist = ((cx-expected_cx)**2 + (cy-expected_cy)**2)**0.5
                if dist < max(slw, slh)*3.0 and gap <= self.cfg.oru_max_gap:
                    self._tracker = self._saved_tracker
                    if gap >= 2: self._tracker.oru_reupdate(self._saved_last_obs, (cx,cy,w,h), gap)
                    self._tracker.update(cx, cy, w, h)
                    self._obs_history.clear(); self._obs_history.append((cx,cy,w,h))
                    self._lock_hits = self._tracker.hits
                else:
                    self._saved_tracker = None; self._saved_last_obs = (0,0,0,0); self._saved_loss_frame = -1
                    self._tracker = None
            else:
                self._tracker = None
            if self._tracker is None:
                self._tracker = KalmanBoxTracker(cx, cy, w, h, self.cfg.kalman_process_noise, self.cfg.kalman_measurement_noise)
                self._obs_history.clear(); self._obs_history.append((cx,cy,w,h)); self._lock_hits = 1
            self._locked_cls = cls; self._locked_aspect = w/h if h>0 else 1.0
            self._locked_box = (cx,cy,w,h)
            self._conf_ewma = conf; self._lost_frames = 0
            self.state = SOTState.LOCKING

    def _transition_locking(self, dets):
        if self._tracker is None: self.state = SOTState.SEARCHING; return
        self._tracker.predict(); pcx, pcy, pw, ph = self._tracker.box
        best_det, best_iou = None, 0.0
        for det in dets:
            _, _, cx, cy, w, h = det
            iou_val = _iou_expand(pcx, pcy, pw, ph, cx, cy, w, h,
                                 self.cfg.iou_threshold, self.cfg.iou_expand_margin)
            if iou_val > best_iou: best_iou = iou_val; best_det = det
        if best_det is not None and best_iou > self.cfg.iou_threshold:
            cls, conf, cx, cy, w, h = best_det
            self._tracker.update(cx, cy, w, h); self._lock_hits += 1
            self._locked_cls = cls; self._locked_aspect = w/h if h>0 else 1.0
            self._locked_box = (cx,cy,w,h)
            self._conf_ewma = self._conf_ewma_alpha*conf + (1-self._conf_ewma_alpha)*self._conf_ewma
            self._lost_frames = 0
            if self._lock_hits >= self.cfg.min_hits:
                self._tracker.hits = self._lock_hits; self._next_id += 1
                self.state = SOTState.TRACKING
        else:
            self._lost_frames += 1
            if self._lost_frames >= self.cfg.lock_timeout_frames:
                self._tracker = None; self._best_candidate = None; self._lock_consistent = 0
                self.state = SOTState.SEARCHING

    def _transition_tracking(self, dets):
        """TRACKING: Kalman predict + expanded IoU match + correct (C++ v5 logic)."""
        if self._tracker is None: self.state = SOTState.SEARCHING; return
        pcx, pcy, pw, ph = self._tracker.predict()

        # Primary: expanded IoU (center-distance override for stickiness)
        best_det, best_val = None, 0.0
        for det in dets:
            _, _, cx, cy, w, h = det
            v = _iou_expand(pcx, pcy, pw, ph, cx, cy, w, h,
                           self.cfg.iou_threshold, self.cfg.iou_expand_margin)
            if v > best_val: best_val = v; best_det = det

        if best_det is not None and best_val > self.cfg.iou_threshold:
            cls, conf, cx, cy, w, h = best_det
            self._tracker.update(cx, cy, w, h)
            self._locked_cls = cls; self._locked_box = (cx,cy,w,h)
            self._locked_aspect = w/h if h>0 else 1.0
            self._conf_ewma = self._conf_ewma_alpha*conf + (1-self._conf_ewma_alpha)*self._conf_ewma
            self._lost_frames = 0; self._obs_history.append((cx,cy,w,h))
            # Wrong-target checks
            anomaly = False
            if self._conf_ewma < self.cfg.conf_floor:
                self._conf_drop_count += 1
                if self._conf_drop_count >= self.cfg.conf_drop_frames: anomaly = True
            else: self._conf_drop_count = 0
            if cls != self._locked_cls:
                self._class_mismatch_count += 1
                if self._class_mismatch_count >= self.cfg.class_mismatch_frames: anomaly = True
            else: self._class_mismatch_count = 0
            cur_aspect = w/h if h>0 else 1.0
            if self._locked_aspect > 0 and max(cur_aspect/self._locked_aspect, self._locked_aspect/cur_aspect) > self.cfg.aspect_drift_factor:
                anomaly = True
            if anomaly: self._verify_clear_count = 0; self.state = SOTState.VERIFYING
            return

        # OCR: last-observation recovery with expanded IoU
        if self.cfg.ocr_enabled and dets and self._locked_box != (0,0,0,0):
            lx, ly, lw, lh = self._locked_box
            for det in dets:
                _, conf, cx, cy, w, h = det
                if _iou_expand(lx, ly, lw, lh, cx, cy, w, h,
                               self.cfg.iou_threshold, self.cfg.iou_expand_margin) > self.cfg.iou_threshold:
                    self._tracker.update(cx, cy, w, h)
                    self._locked_box = (cx,cy,w,h); self._locked_aspect = w/h if h>0 else 1.0
                    self._conf_ewma = self._conf_ewma_alpha*conf + (1-self._conf_ewma_alpha)*self._conf_ewma
                    self._lost_frames = 0; self._obs_history.append((cx,cy,w,h)); return

        # Center-proximity fallback with aspect check
        if dets:
            nearest = min(dets, key=lambda d: (d[2]-pcx)**2+(d[3]-pcy)**2)
            ncx, ncy, nw, nh = nearest[2], nearest[3], nearest[4], nearest[5]
            if ((ncx-pcx)**2+(ncy-pcy)**2)**0.5 < max(pw, ph) * 1.0:
                cur_aspect = nw/nh if nh>0 else 1.0
                ref = self._locked_aspect
                if ref > 0 and max(cur_aspect/ref, ref/cur_aspect) < self.cfg.aspect_drift_factor:
                    self._tracker.update(ncx, ncy, nw, nh)
                    self._locked_box = (ncx,ncy,nw,nh)
                    self._locked_cls = nearest[0]
                    self._conf_ewma = self._conf_ewma_alpha*nearest[1] + (1-self._conf_ewma_alpha)*self._conf_ewma
                    self._lost_frames = 0; self._obs_history.append((ncx,ncy,nw,nh)); return

        # Lost
        self._lost_frames += 1; self._conf_drop_count = 0; self._class_mismatch_count = 0
        if self._lost_frames >= self.cfg.max_age:
            if self.cfg.oru_enabled:
                self._saved_tracker = self._tracker
                self._saved_last_obs = self._locked_box; self._saved_loss_frame = self._lost_frames
            self._tracker = None; self._best_candidate = None; self._lock_consistent = 0
            self.state = SOTState.SEARCHING

    def _transition_verifying(self, dets):
        if self._tracker is None: self.state = SOTState.SEARCHING; return
        pcx, pcy, pw, ph = self._tracker.predict()
        best_det, best_iou = None, 0.0
        for det in dets:
            _, _, cx, cy, w, h = det
            iou_val = _iou_expand(pcx, pcy, pw, ph, cx, cy, w, h,
                                 self.cfg.iou_threshold, self.cfg.iou_expand_margin)
            if iou_val > best_iou: best_iou = iou_val; best_det = det
        if best_det is not None and best_iou > self.cfg.iou_threshold:
            cls, conf, cx, cy, w, h = best_det
            self._tracker.update(cx, cy, w, h)
            self._conf_ewma = self._conf_ewma_alpha*conf + (1-self._conf_ewma_alpha)*self._conf_ewma
            cleared = True
            if self._conf_ewma < self.cfg.conf_floor: cleared = False
            if cls != self._locked_cls: cleared = False
            cur_aspect = w/h if h>0 else 1.0
            if self._locked_aspect > 0 and max(cur_aspect/self._locked_aspect, self._locked_aspect/cur_aspect) > self.cfg.aspect_drift_factor:
                cleared = False
            if cleared:
                self._verify_clear_count += 1
                if self._verify_clear_count >= self.cfg.verify_clear_frames:
                    self._verify_clear_count = 0; self._conf_drop_count = 0; self._class_mismatch_count = 0
                    self.state = SOTState.TRACKING
            else:
                self._verify_clear_count = 0
                alt = self._find_alternate(dets)
                if alt is not None: self._start_switch(alt)
        else:
            self._lost_frames += 1
            alt = self._find_alternate(dets)
            if alt is not None: self._start_switch(alt)
            elif self._lost_frames >= self.cfg.max_age:
                self._tracker = None; self._best_candidate = None; self._lock_consistent = 0
                self.state = SOTState.SEARCHING

    def _transition_switching(self, dets):
        if self._new_tracker is None: self.state = SOTState.SEARCHING; return
        ncx, ncy, nw, nh = self._new_tracker.predict()
        best_det, best_iou = None, 0.0
        for det in dets:
            _, _, cx, cy, w, h = det
            iou_val = _iou_expand(ncx, ncy, nw, nh, cx, cy, w, h,
                                 self.cfg.iou_threshold, self.cfg.iou_expand_margin)
            if iou_val > best_iou: best_iou = iou_val; best_det = det
        if best_det is not None and best_iou > self.cfg.iou_threshold:
            _, _, cx, cy, w, h = best_det
            self._new_tracker.update(cx, cy, w, h)
        if self._tracker is not None: self._old_last_pos = self._tracker.predict_only()
        self._switch_frame += 1
        if self._switch_frame >= self.cfg.blend_frames:
            self._tracker = self._new_tracker; self._new_tracker = None; self._switch_frame = 0
            if self._tracker:
                _, _, tw, th = self._tracker.box
                self._locked_aspect = tw/th if th>0 else 1.0
                self._lost_frames = 0; self._conf_drop_count = 0; self._class_mismatch_count = 0; self._verify_clear_count = 0
            self.state = SOTState.TRACKING

    def _find_alternate(self, dets):
        if not dets: return None
        best, best_score = None, 0.0
        lock_cx, lock_cy, lock_w, lock_h = self._locked_box
        for cls, conf, cx, cy, w, h in dets:
            if _iou(lock_cx, lock_cy, lock_w, lock_h, cx, cy, w, h) > 0.3: continue
            s = _score_detection(cls, conf, cx, cy, w*h, self.cfg)
            if s > best_score and s >= self.cfg.switch_score_threshold: best_score = s; best = (cls, conf, cx, cy, w, h)
        return best

    def _start_switch(self, alt):
        cls, conf, cx, cy, w, h = alt
        self._new_tracker = KalmanBoxTracker(cx, cy, w, h, self.cfg.kalman_process_noise, self.cfg.kalman_measurement_noise)
        if self._tracker is not None: self._old_last_pos = self._tracker.box
        self._switch_frame = 0; self._next_id += 1; self.state = SOTState.SWITCHING

    def _produce_output(self):
        if self.state == SOTState.SEARCHING: return None
        if self.state == SOTState.SWITCHING and self._new_tracker is not None:
            t = self._switch_frame / max(self.cfg.blend_frames, 1); alpha = _smoothstep(t)
            ocx, ocy, ow, oh = self._old_last_pos; ncx, ncy, nw, nh = self._new_tracker.box
            return TrackedTarget(cx=ocx+alpha*(ncx-ocx), cy=ocy+alpha*(ncy-ocy),
                w=ow+alpha*(nw-ow), h=oh+alpha*(nh-oh), conf=0.5, cls=self._locked_cls,
                track_id=self._next_id, state=SOTState.SWITCHING, is_blending=True, blend_progress=alpha)
        if self._tracker is not None:
            tcx, tcy, tw, th = self._tracker.box
            # Damp: freeze toward last observation when lost (match C++ v5)
            lost = self._tracker.time_since_update
            if lost > 0 and self._locked_box != (0, 0, 0, 0):
                alpha = min(1.0, lost / 10.0)  # fully frozen after 10 lost frames
                lx, ly, lw, lh = self._locked_box
                tcx = lx + (tcx - lx) * (1.0 - alpha)
                tcy = ly + (tcy - ly) * (1.0 - alpha)
                tw  = lw + (tw  - lw) * (1.0 - alpha)
                th  = lh + (th  - lh) * (1.0 - alpha)
            return TrackedTarget(cx=tcx, cy=tcy, w=tw, h=th, conf=self._conf_ewma,
                cls=self._locked_cls, track_id=self._next_id if self._next_id>0 else 0,
                state=self.state, is_blending=False, blend_progress=0.0)
        return None
