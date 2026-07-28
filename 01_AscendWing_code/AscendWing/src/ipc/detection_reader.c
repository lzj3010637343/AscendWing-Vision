/**
 * detection_reader.c — AscendCL/POSIX 共享队列 → JSON 行输出到 stdout
 * 用法: ./detection_reader [--benchmark]
 */

#include "ascend_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

static volatile int g_running = 1;
static void sig_handler(int sig) { (void)sig; g_running = 0; }

int main(int argc, char **argv) {
    int benchmark = 0, poll_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) benchmark = 1;
        if (strcmp(argv[i], "--poll") == 0) poll_mode = 1;
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    aclShmBenchmarkEnable(1);  /* always show IPC latency */
    aclShmBenchmarkSetLabel("det");

    if (aclShmReaderInit(ASHM_DETECTION_Q_NAME) != ASHM_OK) {
        fprintf(stderr, "[det] Init failed (is tracker running?)\n");
        return 1;
    }
    fprintf(stderr, "[det] Ready%s\n", poll_mode ? " (poll 1ms)" : "");

    AShmDetectionFrame frame;
    uint32_t frame_cnt = 0;
    uint64_t last_log_us = 0;

    while (g_running) {
        int ret = aclShmGetDetections(&frame, poll_mode ? 1 : 500);
        if (ret == ASHM_TIMEOUT) continue;
        if (ret != ASHM_OK) { if (g_running) fprintf(stderr, "[det] Read error\n"); break; }

        /* JSON 输出 */
        fprintf(stdout, "[");
        for (uint32_t i = 0; i < frame.num_detections && i < ASHM_MAX_DETECTIONS; i++) {
            const AShmDetection *d = &frame.detections[i];
            fprintf(stdout, "%s{\"cls\":%d,\"conf\":%.4f,\"x\":%.2f,\"y\":%.2f,\"w\":%.2f,\"h\":%.2f}",
                    i > 0 ? "," : "", d->class_id, d->confidence, d->x, d->y, d->width, d->height);
        }
        fprintf(stdout, "]\n");
        fflush(stdout);

        frame_cnt++;

        /* 5 帧汇总 */
        if (benchmark && frame_cnt % 5 == 1) {
            struct timeval tv; gettimeofday(&tv, NULL);
            uint64_t now = tv.tv_sec*1000000ULL + tv.tv_usec;
            double elapsed = (now - last_log_us) / 1000000.0;
            last_log_us = now;
            fprintf(stderr, "[det] #%u | %u objs | %.1f fps (5fr %.0fms, ~%.0fms/fr)\n",
                frame.frame_id, frame.num_detections,
                (elapsed > 0 ? 5.0 / elapsed : 0), elapsed * 1000.0,
                elapsed * 200.0);
        }
    }

    aclShmReaderCleanup();
    fprintf(stderr, "[det] Stopped (%u frames)\n", frame_cnt);
    return 0;
}
