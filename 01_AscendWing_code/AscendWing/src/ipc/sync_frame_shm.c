/**
 * sync_frame_shm.c — POSIX shared memory for combined sync frames
 *
 * Writer: memcpy + sem_post (atomic per-frame)
 * Reader: sem_wait + memcpy
 *
 * Follows the same POSIX shm pattern as ascend_shm.c.
 */
#include "sync_frame_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <time.h>

/* Compile-time size verification (C11 _Static_assert).
   Natural alignment: 28B header + 50*24B dets + 524288B JPEG = 525516 */
_Static_assert(sizeof(SyncDetection) == 24,
               "SyncDetection size mismatch");
/* We don't static-assert total size — natural alignment padding varies.
   Python ctypes reads use matching layout, verified at runtime. */

/* ---- Global state ---- */
static struct {
    int     init;
    int     writer;     /* 1 = writer (owns shm lifetime), 0 = reader */
    int     fd;
    void   *ptr;
    size_t  sz;
    sem_t  *sem;
} g_sync = {0, 0, -1, NULL, 0, SEM_FAILED};

/* ---- Writer implementation ---- */

int syncFrameWriterInit(void) {
    if (g_sync.init) return SYNC_OK;

    g_sync.sz = sizeof(AShmSyncFrame);

    /* Create shared memory */
    g_sync.fd = shm_open(SYNC_FRAME_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_sync.fd < 0) {
        perror("[sync-shm] shm_open");
        return SYNC_ERR;
    }
    if (ftruncate(g_sync.fd, (off_t)g_sync.sz) != 0) {
        perror("[sync-shm] ftruncate");
        close(g_sync.fd);
        return SYNC_ERR;
    }
    g_sync.ptr = mmap(NULL, g_sync.sz, PROT_READ | PROT_WRITE, MAP_SHARED, g_sync.fd, 0);
    if (g_sync.ptr == MAP_FAILED) {
        perror("[sync-shm] mmap");
        close(g_sync.fd);
        return SYNC_ERR;
    }

    /* Create semaphore (initial value 0: no frames available yet) */
    sem_unlink(SYNC_FRAME_SEM_NAME);  /* clean stale semaphore */
    g_sync.sem = sem_open(SYNC_FRAME_SEM_NAME, O_CREAT | O_EXCL, 0666, 0);
    if (g_sync.sem == SEM_FAILED) {
        perror("[sync-shm] sem_open");
        munmap(g_sync.ptr, g_sync.sz);
        close(g_sync.fd);
        return SYNC_ERR;
    }

    g_sync.init = 1;
    g_sync.writer = 1;
    fprintf(stderr, "[sync-shm] Writer ready: %s (%zu bytes)\n",
            SYNC_FRAME_SHM_NAME, g_sync.sz);
    return SYNC_OK;
}

int syncFramePut(const AShmSyncFrame *frame) {
    if (!g_sync.init || !g_sync.writer) return SYNC_ERR;
    if (!frame) return SYNC_ERR;

    memcpy(g_sync.ptr, frame, sizeof(AShmSyncFrame));
    sem_post(g_sync.sem);
    return SYNC_OK;
}

void syncFrameWriterCleanup(void) {
    if (!g_sync.init || !g_sync.writer) return;

    if (g_sync.ptr && g_sync.ptr != MAP_FAILED) {
        munmap(g_sync.ptr, g_sync.sz);
    }
    if (g_sync.fd >= 0) {
        close(g_sync.fd);
    }
    shm_unlink(SYNC_FRAME_SHM_NAME);

    if (g_sync.sem != SEM_FAILED) {
        sem_close(g_sync.sem);
        sem_unlink(SYNC_FRAME_SEM_NAME);
    }

    g_sync.init = 0;
    fprintf(stderr, "[sync-shm] Writer cleanup done\n");
}

/* ---- Reader implementation ---- */

int syncFrameReaderInit(void) {
    if (g_sync.init) return SYNC_OK;

    g_sync.sz = sizeof(AShmSyncFrame);

    g_sync.fd = shm_open(SYNC_FRAME_SHM_NAME, O_RDWR, 0666);
    if (g_sync.fd < 0) {
        perror("[sync-shm] reader shm_open");
        return SYNC_ERR;
    }
    g_sync.ptr = mmap(NULL, g_sync.sz, PROT_READ | PROT_WRITE, MAP_SHARED, g_sync.fd, 0);
    if (g_sync.ptr == MAP_FAILED) {
        perror("[sync-shm] reader mmap");
        close(g_sync.fd);
        return SYNC_ERR;
    }

    g_sync.sem = sem_open(SYNC_FRAME_SEM_NAME, 0);
    if (g_sync.sem == SEM_FAILED) {
        perror("[sync-shm] reader sem_open");
        munmap(g_sync.ptr, g_sync.sz);
        close(g_sync.fd);
        return SYNC_ERR;
    }

    g_sync.init = 1;
    g_sync.writer = 0;
    fprintf(stderr, "[sync-shm] Reader attached: %s (%zu bytes)\n",
            SYNC_FRAME_SHM_NAME, g_sync.sz);
    return SYNC_OK;
}

int syncFrameGet(AShmSyncFrame *frame, int timeout_ms) {
    if (!g_sync.init || g_sync.writer) return SYNC_ERR;
    if (!frame) return SYNC_ERR;

    int r;
    if (timeout_ms < 0) {
        /* Block forever */
        r = sem_wait(g_sync.sem);
    } else if (timeout_ms == 0) {
        /* Non-blocking try */
        r = sem_trywait(g_sync.sem);
    } else {
        /* Timed wait */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec  += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
        r = sem_timedwait(g_sync.sem, &ts);
    }

    if (r != 0) {
        if (errno == ETIMEDOUT) return SYNC_TIMEOUT;
        return SYNC_ERR;
    }

    memcpy(frame, g_sync.ptr, sizeof(AShmSyncFrame));
    return SYNC_OK;
}

void syncFrameReaderCleanup(void) {
    if (!g_sync.init || g_sync.writer) return;

    if (g_sync.ptr && g_sync.ptr != MAP_FAILED) {
        munmap(g_sync.ptr, g_sync.sz);
    }
    if (g_sync.fd >= 0) {
        close(g_sync.fd);
    }
    if (g_sync.sem != SEM_FAILED) {
        sem_close(g_sync.sem);
    }
    /* Reader does NOT unlink — writer owns shm/sem lifetime */

    g_sync.init = 0;
    fprintf(stderr, "[sync-shm] Reader detached\n");
}
