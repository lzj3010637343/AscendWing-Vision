/**
 * image_reader.c — POSIX 共享队列 → length-prefixed JPEG 输出到 stdout
 * 用法: ./image_reader [--benchmark]
 */

#include "ascend_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <arpa/inet.h>

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

    aclShmBenchmarkEnable(1); aclShmBenchmarkSetLabel("img");
    if (aclShmReaderInit(ASHM_IMAGE_Q_NAME) != ASHM_OK) {
        fprintf(stderr, "[img] Init failed (is tracker running?)\n");
        return 1;
    }
    fprintf(stderr, "[img] Ready%s\n", poll_mode ? " (poll 1ms)" : "");

    uint8_t jpeg_buf[ASHM_MAX_JPEG_SIZE];
    uint32_t frame_cnt = 0;
    uint64_t last_log_us = 0;

    while (g_running) {
        uint32_t jpeg_size = 0;
        int ret = aclShmGetImage(jpeg_buf, sizeof(jpeg_buf), &jpeg_size, poll_mode ? 1 : 500);
        if (ret == ASHM_TIMEOUT) continue;
        if (ret != ASHM_OK) { if (g_running) fprintf(stderr, "[img] Read error\n"); break; }
        if (jpeg_size == 0) continue;

        /* length-prefixed binary: 4B network-order size + JPEG */
        uint32_t net_len = htonl(jpeg_size);
        fwrite(&net_len, sizeof(net_len), 1, stdout);
        fwrite(jpeg_buf, 1, jpeg_size, stdout);
        fflush(stdout);

        frame_cnt++;
        if (benchmark && frame_cnt % 5 == 1) {
            struct timeval tv; gettimeofday(&tv, NULL);
            uint64_t now = tv.tv_sec*1000000ULL + tv.tv_usec;
            double elapsed = (now - last_log_us) / 1000000.0;
            last_log_us = now;
            fprintf(stderr, "[img] #%u | %u B | %.1f fps (~%.0fms/fr)\n",
                    frame_cnt, jpeg_size,
                    (elapsed > 0 ? 5.0 / elapsed : 0),
                    elapsed * 200.0);
        }
    }

    aclShmReaderCleanup();
    fprintf(stderr, "[img] Stopped (%u frames)\n", frame_cnt);
    return 0;
}
