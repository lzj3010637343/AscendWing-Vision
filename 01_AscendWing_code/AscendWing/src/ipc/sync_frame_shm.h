#ifndef SYNC_FRAME_SHM_H
#define SYNC_FRAME_SHM_H
/**
 * sync_frame_shm.h — Combined sync frame IPC for AscendWing multi-process.
 *
 * Single /sync_frame_shm bundles detections + raw JPEG atomically
 * (POSIX shm + sem). Layout matches Python ctypes AShmSyncFrame exactly.
 *
 * Also defines SOTControlData for the downstream control path.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASHM_MAX_DETECTIONS 50
#define ASHM_MAX_JPEG_SIZE  (512 * 1024)

/* ---- Detection item (24 bytes, matches Python ctypes) ---- */
typedef struct {
    int32_t class_id;
    float   confidence;
    float   x, y, width, height;
} SyncDetection;

/* ---- Combined sync frame — layout matches Python AShmSyncFrame ----
   Natural alignment (no pragma pack): frame_id(4) + pad4 + ts_ns(8) + num(4) + jpg_sz(4) + padding(4)
   = 28 byte header, then dets[50]*24=1200, then jpeg[524288].
   Total: 28 + 1200 + 524288 = 525516 bytes */
#define SYNC_FRAME_SHM_NAME "/sync_frame_shm"
#define SYNC_FRAME_SEM_NAME "/sync_frame_sem"

typedef struct {
    uint32_t      frame_id;
    uint64_t      timestamp_ns;
    uint32_t      num_detections;
    uint32_t      jpeg_size;
    int32_t       padding;          /* unused, for ctypes layout compat */
    SyncDetection detections[ASHM_MAX_DETECTIONS];
    uint8_t       jpeg_data[ASHM_MAX_JPEG_SIZE];
} AShmSyncFrame;

/* ---- SOT control output (sot_live -> control) ---- */
#define SOT_CONTROL_SHM_NAME "/sot_control_shm"

typedef struct {
    uint32_t seqnum;
    uint32_t frame_id;
    float    cx, cy, w, h;
    float    conf;
    int32_t  track_id;
    int32_t  state;
    uint8_t  has_target;
    uint8_t  _pad[3];
} SOTControlData;  /* 40 bytes packed */

/* ---- Return codes ---- */
#define SYNC_OK       0
#define SYNC_ERR     -1
#define SYNC_TIMEOUT -2

/* Writer API (C++ tracker side) */
int  syncFrameWriterInit(void);
int  syncFramePut(const AShmSyncFrame *frame);
void syncFrameWriterCleanup(void);

/* Reader API (for C consumers; Python uses mmap directly) */
int  syncFrameReaderInit(void);
int  syncFrameGet(AShmSyncFrame *frame, int timeout_ms);
void syncFrameReaderCleanup(void);

#ifdef __cplusplus
}
#endif
#endif
