#ifndef ASCEND_SHM_H
#define ASCEND_SHM_H
/**
 * ascend_shm.h — AscendCL 共享队列 IPC 封装
 *
 * 对 TDT (acltdtQueue + acltdtBuf) 做最小封装，提供生产者/消费者模式：
 *   - 生产者进程: aclShmWriterInit → aclShmPut* → aclShmCleanup
 *   - 消费者进程: aclShmReaderInit → aclShmGet → aclShmCleanup
 *
 * 每个进程只需调用一次 aclInit/aclrtSetDevice（由本库处理），
 * 消费者通过队列名称 attach 到生产者创建的队列。
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 返回值 ---- */
#define ASHM_OK       0
#define ASHM_ERR     -1
#define ASHM_TIMEOUT -2

/* ---- 队列配置 ---- */
#define ASHM_DETECTION_Q_NAME  "detection_q"
#define ASHM_IMAGE_Q_NAME      "image_q"
#define ASHM_QUEUE_DEPTH       4
#define ASHM_DEFAULT_TIMEOUT_MS  (-1)   /* 阻塞等待 */
#define ASHM_MAX_JPEG_SIZE      (512 * 1024)

/* ---- 单个检测项（与 shm_layout.h 对齐） ---- */
typedef struct {
    int32_t class_id;
    float   confidence;
    float   x, y, width, height;
} AShmDetection;

/* ---- 检测帧 ---- */
#define ASHM_MAX_DETECTIONS 50

typedef struct {
    uint32_t      frame_id;
    uint64_t      timestamp_ns;
    uint32_t      num_detections;
    int32_t       padding;
    AShmDetection detections[ASHM_MAX_DETECTIONS];
} AShmDetectionFrame;

/* ---- 写入端 API ---- */

/** 初始化写入端：aclInit + aclrtSetDevice + 创建两个共享队列 */
int aclShmWriterInit(void);

/** 写入一帧检测结果到共享队列（帧内拷贝） */
int aclShmPutDetections(const AShmDetectionFrame *frame);

/** 写入一帧 JPEG 图像到共享队列（帧内拷贝） */
int aclShmPutImage(const uint8_t *jpeg_data, uint32_t jpeg_size);

/** 释放写入端资源 */
void aclShmWriterCleanup(void);

/* ---- 读取端 API ---- */

/**
 * 初始化读取端：aclInit + aclrtSetDevice + attach 到命名队列
 * @param queue_name  ASHM_DETECTION_Q_NAME 或 ASHM_IMAGE_Q_NAME
 * @return ASHM_OK / ASHM_ERR
 */
int aclShmReaderInit(const char *queue_name);

/**
 * 获取一帧检测结果（阻塞直到有数据）
 * @param frame [OUT] 存放检测帧
 * @param timeout_ms  -1 为阻塞等待
 * @return ASHM_OK / ASHM_TIMEOUT / ASHM_ERR
 */
int aclShmGetDetections(AShmDetectionFrame *frame, int timeout_ms);

/**
 * 获取一帧 JPEG 图像
 * @param buf       [OUT] 存放 JPEG 数据的缓冲区（调用者预分配）
 * @param buf_size  [IN]  buf 的最大容量
 * @param jpeg_size [OUT] 实际 JPEG 数据大小
 * @param timeout_ms      -1 为阻塞等待
 * @return ASHM_OK / ASHM_TIMEOUT / ASHM_ERR
 */
int aclShmGetImage(uint8_t *buf, uint32_t buf_size, uint32_t *jpeg_size, int timeout_ms);

/** 释放读取端资源 */
void aclShmReaderCleanup(void);

/* ---- Benchmark (optional) ---- */

/** 启用/禁用 benchmark 日志（默认禁用，stderr 输出） */
void aclShmBenchmarkEnable(int enable);

/** 设置 benchmark 日志的进程名标签 */
void aclShmBenchmarkSetLabel(const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ASCEND_SHM_H */
