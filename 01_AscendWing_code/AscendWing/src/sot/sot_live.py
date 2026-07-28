#!/usr/bin/env python3
"""
sot_live.py — SOT real-time visualization (v2: sync frame, zero desync)
=======================================================================
Direct mmap read of POSIX sync frame shm (/sync_frame_shm + /sync_frame_sem).
Single semaphore guarantees atomic detection+JPEG per frame.

Usage:
  Terminal 1: ./run.sh unified
  Terminal 2: python3 src/sot/sot_live.py
  Browser: http://IP:8081/

Legend:
  Green thick box + label = SOT locked target
  Red thin box            = YOLO raw detections
"""

import argparse, ctypes, errno, mmap, os, queue, select, signal, sys, threading, time
import multiprocessing
from http.server import HTTPServer, BaseHTTPRequestHandler
from types import SimpleNamespace

import cv2
import numpy as np

_cur = os.path.dirname(os.path.abspath(__file__))
if _cur not in sys.path:
    sys.path.insert(0, _cur)
from single_object_tracker import SingleObjectTracker, SOTConfig, SOTState

# ============================================================
#  C struct layouts (must match sync_frame_shm.h)
# ============================================================

class AShmDetection(ctypes.Structure):
    _fields_ = [
        ("class_id",   ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("x",          ctypes.c_float),
        ("y",          ctypes.c_float),
        ("width",      ctypes.c_float),
        ("height",     ctypes.c_float),
    ]

ASHM_MAX_DETECTIONS = 50
ASHM_MAX_JPEG_SIZE  = 512 * 1024

class AShmSyncFrame(ctypes.Structure):
    _fields_ = [
        ("frame_id",       ctypes.c_uint32),
        ("timestamp_ns",   ctypes.c_uint64),
        ("num_detections", ctypes.c_uint32),
        ("jpeg_size",      ctypes.c_uint32),
        ("padding",        ctypes.c_int32),
        ("detections",     AShmDetection * ASHM_MAX_DETECTIONS),
        ("jpeg_data",      ctypes.c_uint8 * ASHM_MAX_JPEG_SIZE),
    ]

SHM_NAME = "/sync_frame_shm"
SEM_NAME = b"/sync_frame_sem"
SHM_SIZE = ctypes.sizeof(AShmSyncFrame)

# ---- SOT control output (→ downstream consumer) ----
class SOTControlData(ctypes.Structure):
    _fields_ = [
        ("seqnum",    ctypes.c_uint32),
        ("frame_id",  ctypes.c_uint32),
        ("cx",        ctypes.c_float),
        ("cy",        ctypes.c_float),
        ("w",         ctypes.c_float),
        ("h",         ctypes.c_float),
        ("conf",      ctypes.c_float),
        ("track_id",  ctypes.c_int32),
        ("state",     ctypes.c_int32),
        ("has_target",ctypes.c_uint8),
        ("_pad",      ctypes.c_uint8 * 3),
    ]

SOT_SHM_PATH = "/dev/shm/sot_control_shm"
SOT_SHM_SIZE = ctypes.sizeof(SOTControlData)
_sot_fd = None
_sot_mm = None
_sot_seq = 0

def sot_shm_init():
    global _sot_fd, _sot_mm
    try:
        _sot_fd = os.open(SOT_SHM_PATH, os.O_RDWR | os.O_CREAT, 0o666)
        os.ftruncate(_sot_fd, SOT_SHM_SIZE)
        _sot_mm = mmap.mmap(_sot_fd, SOT_SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
    except Exception as e:
        print(f"[SOT] WARNING: sot shm init failed: {e}")

def sot_shm_put(target, sot_state_str):
    global _sot_seq
    if _sot_mm is None:
        return
    _sot_seq += 1
    d = SOTControlData()
    d.seqnum   = _sot_seq
    d.frame_id = _sot_seq
    state_map = {"SEARCHING":0,"LOCKING":1,"TRACKING":2,"MEMORY":3,"SWITCHING":1,"VERIFYING":1}
    if target is not None:
        d.cx, d.cy, d.w, d.h = target.cx, target.cy, target.w, target.h
        d.conf      = target.conf
        d.track_id  = target.track_id
        d.state     = state_map.get(sot_state_str, 2)
        d.has_target = 1
    else:
        d.state      = 0
        d.has_target = 0
    _sot_mm.seek(0)
    _sot_mm.write(bytes(d))
    _sot_mm.flush()

def sot_shm_close():
    global _sot_mm, _sot_fd
    if _sot_mm:
        _sot_mm.close()
    if _sot_fd is not None:
        os.close(_sot_fd)

# ============================================================
#  POSIX semaphore via libc
# ============================================================

class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

_libc = ctypes.CDLL("libc.so.6")
_libc.sem_open.restype   = ctypes.c_void_p
_libc.sem_wait.argtypes  = [ctypes.c_void_p]
_libc.sem_wait.restype   = ctypes.c_int
_libc.sem_post.argtypes  = [ctypes.c_void_p]
_libc.sem_post.restype   = ctypes.c_int
_libc.sem_close.argtypes = [ctypes.c_void_p]
_libc.sem_close.restype  = ctypes.c_int

def shm_reader_thread(sem, buf, frame_queue, running):
    """Daemon: block on sem_wait, copy raw frame into queue.

    Dedup by frame_id: the named semaphore can accumulate/duplicate posts
    (e.g. after a restart while vision kept posting), which would make the
    main loop re-process the same frame many times and pin a CPU core.
    We peek frame_id (first uint32 in the struct) and skip unchanged frames.
    """
    last_fid = -1
    while running.value:
        ret = _libc.sem_wait(sem)
        if ret != 0:
            if running.value: time.sleep(0.001)
            continue
        fid = int.from_bytes(buf[0:4], 'little')
        if fid == last_fid:
            continue   # same frame — ignore spurious/duplicate sem post
        last_fid = fid
        raw = bytes(buf)
        # Drop-old: if queue is full (main loop busy), evict the stale frame so
        # main always wakes on the newest frame — minimizes end-to-end latency.
        try:
            frame_queue.put_nowait(raw)
        except queue.Full:
            try:
                frame_queue.get_nowait()   # discard stale
                frame_queue.put_nowait(raw)
            except queue.Empty:
                pass


def render_process_main(overlay_mpq, port, running):
    """Render process: own CPU core (no GIL contention with SOT main loop).

    Pulls overlay frames from the multiprocessing Queue, draws boxes, encodes
    JPEG, and serves MJPEG over HTTP. Runs independently — if it lags, the
    queue back-pressures and the main loop drops render frames (never blocks
    SOT/shm writes to the downstream consumer). A crash here does not affect the main
    process.
    """
    import signal as _sig
    # Render process: ignore Ctrl+C (parent will terminate us), avoid double
    # signal handlers fighting with the main process.
    _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    # ---- MJPEG HTTP server (lives entirely in this process) ----
    srv = HTTPServer(('0.0.0.0', port), MjpegHandler)

    def _serve():
        try: srv.serve_forever()
        except Exception: pass
    srv_thread = threading.Thread(target=_serve, daemon=True)
    srv_thread.start()

    while running.value:
        try:
            data = overlay_mpq.get(timeout=0.2)
        except queue.Empty:
            continue
        if data is None:        # sentinel — shutdown
            break
        # Drain to latest (drop stale render frames; keep latency low)
        while True:
            try: data = overlay_mpq.get_nowait()
            except queue.Empty: break
            if data is None: break
        if data is None: break
        overlay = draw_sot_overlay(
            data['jpg'], data['target'], data['dets'],
            data['sot_state'], data['frame_id'], data['track_rate'])
        if overlay:             # None on corrupt JPEG → keep last frame
            MjpegHandler.jpeg_data = overlay

    try: srv.shutdown()
    except Exception: pass


# ============================================================
#  MJPEG server
# ============================================================

class MjpegHandler(BaseHTTPRequestHandler):
    jpeg_data = b''
    _active_streams = 0
    _max_streams = 2   # cap concurrent MJPEG streams — each is a tight loop

    def do_GET(self):
        if self.path in ('/', '/stream'):
            # Limit concurrent streams: extra tabs/refreshes get 503 instead
            # of spawning another tight write-loop that burns CPU.
            if MjpegHandler._active_streams >= MjpegHandler._max_streams:
                self.send_response(503); self.end_headers()
                return
            MjpegHandler._active_streams += 1
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                while True:
                    data = MjpegHandler.jpeg_data
                    if data:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(data)}\r\n\r\n'.encode())
                        self.wfile.write(data)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.025)   # ~40fps push; balances smoothness vs CPU
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                MjpegHandler._active_streams -= 1
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *args): pass


# ============================================================
#  Drawing
# ============================================================

STATUS_BAR_H = 32
RENDER_SCALE = 0.5    # downscale factor for overlay render (lower=lighter=faster)
RENDER_JPEG_QUALITY = 35   # MJPEG encode quality (lower=lighter)

def draw_sot_overlay(jpeg_bytes, target, dets, sot_state, frame_num, track_rate):
    """Draw SOT locked target (green thick) + detection boxes (red thin) + status bar."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None   # corrupt JPEG — skip, keep last frame
    h, w = img.shape[:2]
    scale_x = 800.0 / w if w > 0 else 1.0
    scale_y = 600.0 / h if w > 0 else 1.0
    # Downscale for lighter draw+encode (detection coords are in 800x600 space,
    # so reuse scale_x/scale_y which already map to the small render size).
    if RENDER_SCALE != 1.0:
        img = cv2.resize(img, (max(1, int(w*RENDER_SCALE)), max(1, int(h*RENDER_SCALE))),
                         interpolation=cv2.INTER_AREA)
        scale_x = scale_x / RENDER_SCALE
        scale_y = scale_y / RENDER_SCALE
        h, w = img.shape[:2]

    # YOLO detections: red thin
    for d in (dets or []):
        x1 = int(d["x"] / scale_x); y1 = int(d["y"] / scale_y)
        x2 = int((d["x"] + d["w"]) / scale_x)
        y2 = int((d["y"] + d["h"]) / scale_y)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1: continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # SOT locked target: green thick + corner marks
    if target:
        x1 = int((target.cx - target.w / 2) / scale_x)
        y1 = int((target.cy - target.h / 2) / scale_y)
        x2 = int((target.cx + target.w / 2) / scale_x)
        y2 = int((target.cy + target.h / 2) / scale_y)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        corner = min(25, (x2 - x1) // 3, (y2 - y1) // 3)
        cv2.line(img, (x1, y1), (x1 + corner, y1), (0, 255, 0), 6)
        cv2.line(img, (x1, y1), (x1, y1 + corner), (0, 255, 0), 6)
        cv2.line(img, (x2, y1), (x2 - corner, y1), (0, 255, 0), 6)
        cv2.line(img, (x2, y1), (x2, y1 + corner), (0, 255, 0), 6)
        cv2.line(img, (x1, y2), (x1 + corner, y2), (0, 255, 0), 6)
        cv2.line(img, (x1, y2), (x1, y2 - corner), (0, 255, 0), 6)
        cv2.line(img, (x2, y2), (x2 - corner, y2), (0, 255, 0), 6)
        cv2.line(img, (x2, y2), (x2, y2 - corner), (0, 255, 0), 6)

        state_icon = {"TRACKING": "T", "LOCKING": "L", "SWITCHING": "S",
                      "VERIFYING": "V", "SEARCHING": "?", "MEMORY": "M"}.get(sot_state, "?")
        label = f"SOT#{target.track_id} [{state_icon}] c{target.cls} {target.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = y1 - 6 if y1 > th + 10 else y2 + th + 6
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 6, label_y + 4), (0, 180, 0), -1)
        cv2.putText(img, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if target.is_blending:
            bar_w = x2 - x1; bar_h = 5; bar_y = y2 + 8
            cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_h), (80, 80, 80), -1)
            cv2.rectangle(img, (x1, bar_y), (x1 + int(bar_w * target.blend_progress), bar_y + bar_h),
                          (0, 255, 255), -1)

    # Status bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, STATUS_BAR_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    state_colors = {"TRACKING": (0, 255, 0), "LOCKING": (0, 255, 255),
                    "SWITCHING": (255, 255, 0), "VERIFYING": (255, 0, 255),
                    "SEARCHING": (150, 150, 150), "MEMORY": (255, 165, 0)}
    sc = state_colors.get(sot_state, (255, 255, 255))
    cv2.putText(img, f"SOT: {sot_state}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
    right_text = f"F{frame_num} | dets={len(dets) if dets else 0} | rate={track_rate:.0f}%"
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, right_text, (w - rw - 10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, RENDER_JPEG_QUALITY])
    return enc.tobytes()


# ============================================================
#  Main loop
# ============================================================

def main():
    p = argparse.ArgumentParser(description='SOT real-time visualization (sync frame v2)')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--lock-score', type=float, default=0.4)
    p.add_argument('--lock-frames', type=int, default=2)
    p.add_argument('--min-hits', type=int, default=2)
    p.add_argument('--max-age', type=int, default=30)
    p.add_argument('--blend-frames', type=int, default=10)
    p.add_argument('--kalman-pn', type=float, default=0.03)
    p.add_argument('--kalman-mn', type=float, default=0.08)
    p.add_argument('--no-kalman', action='store_true')
    p.add_argument('--target-class', type=int, default=-1)
    # --- stickiness: lock target hard, tolerate low conf / drift ---
    p.add_argument('--min-conf', type=float, default=0.25,
                   help='Min detection conf to enter matching pool (default 0.25, lower=sees weak dets)')
    p.add_argument('--conf-floor', type=float, default=0.35,
                   help='Conf EWMA floor below which target is flagged wrong (default 0.35, lower=stickier)')
    p.add_argument('--conf-drop-frames', type=int, default=5,
                   help='Consecutive frames below conf_floor before anomaly (default 5, higher=stickier)')
    p.add_argument('--aspect-drift', type=float, default=1.8,
                   help='Aspect ratio drift ratio before anomaly (default 1.8, higher=stickier)')
    p.add_argument('--switch-score', type=float, default=0.5,
                   help='Score a candidate must reach to switch target (default 0.5, higher=stickier)')
    p.add_argument('--iou-margin', type=float, default=1.0,
                   help='Expanded-IoU inflation radius margin (default 1.0, higher=stickier vs jitter)')
    args = p.parse_args()

    # ---- SOT control shm ----
    sot_shm_init()

    # ---- SOT ----
    sot = SingleObjectTracker(SOTConfig(
        target_class=args.target_class,
        lock_score_threshold=args.lock_score,
        lock_consistency_frames=args.lock_frames,
        min_hits=args.min_hits,
        max_age=args.max_age,
        blend_frames=args.blend_frames,
        kalman_process_noise=args.kalman_pn,
        kalman_measurement_noise=args.kalman_mn,
        min_confidence=args.min_conf,
        conf_floor=args.conf_floor,
        conf_drop_frames=args.conf_drop_frames,
        aspect_drift_factor=args.aspect_drift,
        switch_score_threshold=args.switch_score,
        iou_expand_margin=args.iou_margin,
    ))

    # ---- Attach to sync frame POSIX shm ----
    shm_path = f"/dev/shm{SHM_NAME}"
    print(f"[SOT] Attaching to {shm_path} ({SHM_SIZE/1024:.0f} KB) ...")

    fd = os.open(shm_path, os.O_RDONLY)
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)

    sem = _libc.sem_open(SEM_NAME, 0)
    if sem is None or ctypes.c_void_p(sem).value is None:
        print("[SOT] ERROR: sem_open failed (is unified tracker running?)")
        buf.close(); return
    print(f"[SOT] Sync shm + sem ready (atomic det+img, single semaphore)")

    # ---- Exit handling ----
    running = multiprocessing.Value('b', True)
    ctrlc_count = [0]

    def _sig(sig, frame):
        ctrlc_count[0] += 1
        running.value = False
        if ctrlc_count[0] >= 2:
            os._exit(0)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ---- Queues ----
    # frame_queue: reader thread -> main loop. maxsize=1 for minimal latency:
    # reader drops stale frames, main always gets the newest — no buffering lag.
    frame_queue = queue.Queue(maxsize=1)
    # overlay_mpq: main loop -> render PROCESS. maxsize=1 so the displayed
    # frame stays ~1 frame behind the tracked target (same-frame alignment),
    # not 2+. When render lags, put_nowait drops — SOT/shm never blocks.
    overlay_mpq = multiprocessing.Queue(maxsize=1)

    reader = threading.Thread(target=shm_reader_thread,
        args=(sem, buf, frame_queue, running), daemon=True, name="shm-reader")
    reader.start()

    # ---- Render process (own core, owns MJPEG server) ----
    renderer = multiprocessing.Process(
        target=render_process_main,
        args=(overlay_mpq, args.port, running), name="render")
    renderer.start()
    print(f"[SOT] MJPEG → http://0.0.0.0:{args.port}/  (render process pid={renderer.pid})")

    print(f"[SOT] Procs: shm-reader(thread) + SOT(main) + render(process)")
    print(f"{'='*55}")
    print(f"  Green thick = SOT locked target")
    print(f"  Red thin    = YOLO detections (same frame, zero delay)")
    print(f"  'T'=Tracking  'M'=Memory  'S'=Switching")
    print(f"  Exit: Ctrl+C (double-click to force kill)")
    print(f"{'='*55}\n")

    # ---- Main loop: SOT only (no rendering, full frame rate) ----
    frame = 0
    total_tracked = 0
    prev_state = None
    last_status = 0
    sf = AShmSyncFrame()
    bench = int(os.environ.get('BENCH', '0'))
    bench_empty = 0

    while running.value:
        t_q0 = time.perf_counter()
        try:
            raw = frame_queue.get(timeout=0.1)
        except queue.Empty:
            if bench:
                bench_empty += 1
                if bench_empty >= 5:      # producer gone -> stop
                    running.value = False
            continue
        t_q1 = time.perf_counter()
        bench_empty = 0

        ctypes.memmove(ctypes.byref(sf), raw, min(len(raw), SHM_SIZE))
        if sf.jpeg_size == 0 or sf.jpeg_size > ASHM_MAX_JPEG_SIZE:
            continue

        # Extract detections
        dets = []
        n_det = min(sf.num_detections, ASHM_MAX_DETECTIONS)
        for i in range(n_det):
            d = sf.detections[i]
            dets.append({"cls": d.class_id, "conf": d.confidence,
                         "x": d.x, "y": d.y, "w": d.width, "h": d.height})

        jpg = bytes(sf.jpeg_data[:sf.jpeg_size])

        # SOT update (~2ms)
        t_s0 = time.perf_counter()
        t = sot.update(dets, time.time())
        t_s1 = time.perf_counter()
        frame += 1
        if t: total_tracked += 1

        # Write SOT result → downstream
        sot_shm_put(t, sot.state.name)

        if bench:
            e2e_us = (time.time_ns() - sf.timestamp_ns) // 1000
            print(f"[BENCH] f={sf.frame_id} wait={(t_q1-t_q0)*1e6:.0f} "
                  f"sot={(t_s1-t_s0)*1e6:.0f} e2e={e2e_us}", flush=True)
            if bench > 0 and frame >= bench:
                running.value = False

        track_rate = total_tracked / frame * 100 if frame > 0 else 0

        # State change logging
        if sot.state != prev_state:
            old = prev_state.name if prev_state else "START"
            print(f"  [{frame:5d}] {old} → {sot.state.name}"
                  f"{'  (switch #%d)' % (total_tracked) if sot.state == SOTState.SWITCHING else ''}")
            prev_state = sot.state

        # Terminal status (1 Hz)
        if time.time() - last_status > 1.0:
            last_status = time.time()
            if t:
                print(f"\r[SOT] {sot.state.name:<10s} id={t.track_id} "
                      f"@({t.cx:6.0f},{t.cy:6.0f}) conf={t.conf:.2f}  "
                      f"rate={track_rate:.0f}%  {'[blend]' if t.is_blending else ''}  "
                      f"F{sf.frame_id}  ", end='', flush=True)
            else:
                print(f"\r[SOT] {sot.state.name:<10s} searching...            "
                      f"rate={track_rate:.0f}%  F{sf.frame_id}  ", end='', flush=True)

        # Ship to render thread
        target_snap = None
        if t:
            target_snap = SimpleNamespace(
                cx=t.cx, cy=t.cy, w=t.w, h=t.h,
                track_id=t.track_id, cls=t.cls, conf=t.conf,
                is_blending=t.is_blending, blend_progress=t.blend_progress)
        try:
            overlay_mpq.put_nowait({
                'jpg': jpg, 'target': target_snap, 'dets': dets,
                'sot_state': sot.state.name, 'frame_id': sf.frame_id,
                'track_rate': track_rate,
            })
        except queue.Full:
            pass   # render lagging — drop this render frame, never block SOT

    # ---- Cleanup ----
    running.value = False
    try:
        overlay_mpq.put_nowait(None)     # sentinel to render process
    except Exception:
        pass
    if sem: _libc.sem_post(sem)
    reader.join(timeout=2.0)
    # Render process: give it a moment for graceful exit, then force.
    renderer.join(timeout=2.0)
    if renderer.is_alive():
        renderer.terminate()
        renderer.join(timeout=1.0)
    if sem: _libc.sem_close(sem)
    sot_shm_close()
    buf.close()
    print(f"\n[SOT] Done. {frame} frames, track rate {total_tracked/frame*100:.1f}%"
          if frame > 0 else "\n[SOT] Done. 0 frames.")


if __name__ == '__main__':
    main()
