
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/device.h>
#include <zephyr/drivers/video.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/atomic.h>

#include <stm32h7xx_hal.h>

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <limits.h>

#pragma GCC push_options
#pragma GCC optimize("O3", "unroll-loops")

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma GCC pop_options

extern "C" {
#include "network.h"
#include "network_data.h"
#include "main.h"
#include "app_postprocess.h"
}
}
LOG_MODULE_REGISTER(camera_uart_test, LOG_LEVEL_INF);

/* =========================================================
 * Configuration
 * ========================================================= */

#define CAM_WIDTH           320
#define CAM_HEIGHT          240
#define CAM_BPP             2

#define AI_WIDTH            192
#define AI_HEIGHT           192
#define AI_CHANNELS         3

#define CAMERA_BUF_COUNT    2
#define PROC_BUF_COUNT      2

#define UART_BAUDRATE       460800U

#define FIXED_PACKET_SIZE   5000U
#define TX_SLOT_COUNT       2

#define MAX_DETECTIONS      10

#define JPEG_QUALITY        50

#define MAX_FRAME_SIZE \
    (CAM_WIDTH * CAM_HEIGHT * CAM_BPP)

#define MAX_JPEG_SIZE \
    (FIXED_PACKET_SIZE - sizeof(frame_header) - \
     MAX_DETECTIONS * sizeof(bbox_data))

#define CAM_THREAD_STACK_SIZE 4096
#define TX_THREAD_STACK_SIZE  4096

#define JPEG_SCRATCH_SIZE 0

/* =========================================================
 * Packet protocol
 * ========================================================= */

#pragma pack(push, 1)

struct frame_header {
    uint16_t magic_word;
    uint32_t image_size;
    uint16_t box_count;
};

struct bbox_data {
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
    float conf;
    uint8_t label_char;
};

#pragma pack(pop)

static_assert(sizeof(frame_header) == 8);
static_assert(sizeof(bbox_data) == 13);

/* =========================================================
 * UART TX buffers
 * ========================================================= */

static const struct device *uart_dev =
    DEVICE_DT_GET(DT_NODELABEL(uart8));

static uint8_t uart_tx_buffer[TX_SLOT_COUNT][FIXED_PACKET_SIZE]
    __attribute__((aligned(32), section("SDRAM2")));

static struct k_sem tx_slot_free[TX_SLOT_COUNT];

struct tx_msg {
    uint8_t slot;

    uint32_t frame_id;
    uint32_t total_len;

    uint32_t prep_us;
    uint32_t ai_us;
    uint32_t post_us;
    uint32_t jpeg_us;

    uint32_t jpeg_size;
    uint16_t box_count;
};

K_MSGQ_DEFINE(tx_msgq, sizeof(struct tx_msg), 2, 4);

/* =========================================================
 * Camera
 * ========================================================= */

static const struct device *video_dev;

static uint8_t raw_video_data[CAMERA_BUF_COUNT][MAX_FRAME_SIZE]
    __attribute__((aligned(32), section("SDRAM2")));

static struct video_buffer video_buf_metadata[CAMERA_BUF_COUNT];

static struct video_format cam_fmt;

/* =========================================================
 * AI
 * ========================================================= */

static AppConfig_TypeDef App_Config;

static postprocess_outBuffer_t detections[MAX_DETECTIONS];

static uint8_t my_camera_data[PROC_BUF_COUNT]
    [STAI_NETWORK_IN_1_SIZE_BYTES]
    __attribute__((aligned(32), section("SDRAM2")));

static uint8_t network_context[STAI_NETWORK_CONTEXT_SIZE]
    __attribute__((aligned(32), section("SDRAM2")));

static stai_network *network =
    (stai_network *)network_context;

static uint8_t my_output_0[STAI_NETWORK_OUT_1_SIZE_BYTES]
    __attribute__((aligned(32), section("SDRAM2")));

static uint8_t my_output_1[STAI_NETWORK_OUT_2_SIZE_BYTES]
    __attribute__((aligned(32), section("SDRAM2")));

static uint8_t my_output_2[STAI_NETWORK_OUT_3_SIZE_BYTES]
    __attribute__((aligned(32), section("SDRAM2")));

static uint8_t my_activations[STAI_NETWORK_ACTIVATIONS_SIZE_BYTES]
    __attribute__((aligned(32)));

static uint8_t *in_buffers[1];

static uint8_t *act_buffers[1] = {
    my_activations
};

static uint8_t *out_buffers[3] = {
    my_output_0,
    my_output_1,
    my_output_2
};

static struct k_sem buf_free_sem[PROC_BUF_COUNT];

/* =========================================================
 * Frame queue
 * ========================================================= */

struct frame_msg {
    int buf_idx;

    uint32_t frame_id;
    uint32_t prep_us;
};

K_MSGQ_DEFINE(frame_msgq, sizeof(frame_msg), 2, 4);

/* =========================================================
 * Threads
 * ========================================================= */

K_THREAD_STACK_DEFINE(
    cam_thread_stack,
    CAM_THREAD_STACK_SIZE
);

K_THREAD_STACK_DEFINE(
    tx_thread_stack,
    TX_THREAD_STACK_SIZE
);

static struct k_thread cam_thread_data;
static struct k_thread tx_thread_data;

/* =========================================================
 * Runtime flags
 * ========================================================= */

static atomic_t ai_running = ATOMIC_INIT(0);
static atomic_t jpeg_running = ATOMIC_INIT(0);

/* =========================================================
 * Timing helpers
 * ========================================================= */

static inline uint32_t cycles_to_us(uint32_t cycles)
{
    return k_cyc_to_us_floor32(cycles);
}

/* =========================================================
 * Statistics
 * ========================================================= */

static volatile uint32_t uart_isr_calls;
static volatile uint32_t uart_bytes_queued;

static volatile uint32_t uart_max_isr_gap_cycles;
static volatile uint32_t uart_last_isr_cycle;

static volatile uint32_t uart_bytes_during_ai;
static volatile uint32_t uart_isr_during_ai;

static uint32_t uart_tx_duration_us;

static uint32_t jpeg_min_us = UINT32_MAX;
static uint32_t jpeg_max_us = 0;
static uint64_t jpeg_total_us = 0;
static uint32_t jpeg_count = 0;

static uint32_t frame_count = 0;
static uint32_t dropped_frames = 0;

/* =========================================================
 * UART interrupt state
 * ========================================================= */

static const uint8_t *uart_tx_ptr = nullptr;

static volatile size_t uart_tx_remaining = 0;

static struct k_sem uart_tx_done_sem;

/* =========================================================
 * UART ISR
 * ========================================================= */

static void uart_isr(
    const struct device *dev,
    void *user_data
)
{
    ARG_UNUSED(user_data);

    if (!uart_irq_update(dev)) {
        return;
    }

    uint32_t now = k_cycle_get_32();

    uart_isr_calls++;

    if (atomic_get(&ai_running)) {
        uart_isr_during_ai++;
    }

    if (uart_last_isr_cycle != 0) {

        uint32_t gap = now - uart_last_isr_cycle;

        if (gap > uart_max_isr_gap_cycles) {
            uart_max_isr_gap_cycles = gap;
        }
    }

    uart_last_isr_cycle = now;

    if (!uart_irq_tx_ready(dev)) {
        return;
    }

    if (uart_tx_remaining > 0) {

        int request_len =
            uart_tx_remaining > INT_MAX
            ? INT_MAX
            : (int)uart_tx_remaining;

        int n = uart_fifo_fill(
            dev,
            uart_tx_ptr,
            request_len
        );

        if (n > 0) {

            uart_tx_ptr += n;
            uart_tx_remaining -= (size_t)n;

            uart_bytes_queued += (uint32_t)n;

            if (atomic_get(&ai_running)) {
                uart_bytes_during_ai += (uint32_t)n;
            }
        }
    }

    if (uart_tx_remaining == 0 &&
        uart_irq_tx_complete(dev)) {

        uart_irq_tx_disable(dev);

        k_sem_give(&uart_tx_done_sem);
    }
}

/* =========================================================
 * UART transmission
 * ========================================================= */

static int uart_send_blocking(
    const uint8_t *data,
    size_t len
)
{
    if (data == nullptr || len != FIXED_PACKET_SIZE) {
        return -EINVAL;
    }

    uart_isr_calls = 0;
    uart_bytes_queued = 0;

    uart_max_isr_gap_cycles = 0;
    uart_last_isr_cycle = 0;

    uart_bytes_during_ai = 0;
    uart_isr_during_ai = 0;

    uart_tx_ptr = data;
    uart_tx_remaining = len;

    uint32_t start = k_cycle_get_32();

    uart_irq_tx_enable(uart_dev);

    int ret = k_sem_take(
        &uart_tx_done_sem,
        K_SECONDS(5)
    );

    uint32_t end = k_cycle_get_32();

    if (ret != 0) {

        uart_irq_tx_disable(uart_dev);

        uart_tx_remaining = 0;

        return -ETIMEDOUT;
    }

    uart_tx_duration_us = cycles_to_us(end - start);

    if (uart_bytes_queued != len) {
        return -EIO;
    }

    return 0;
}

/* =========================================================
 * SDRAM / MPU
 * ========================================================= */

static void Enable_SDRAM_Cache(void)
{
    MPU_Region_InitTypeDef MPU_InitStruct = {0};

    HAL_MPU_Disable();

    MPU_InitStruct.Enable = MPU_REGION_ENABLE;
    MPU_InitStruct.BaseAddress = 0xD0000000;
    MPU_InitStruct.Size = MPU_REGION_SIZE_32MB;

    MPU_InitStruct.AccessPermission =
        MPU_REGION_FULL_ACCESS;

    MPU_InitStruct.IsBufferable =
        MPU_ACCESS_BUFFERABLE;

    MPU_InitStruct.IsCacheable =
        MPU_ACCESS_CACHEABLE;

    MPU_InitStruct.IsShareable =
        MPU_ACCESS_NOT_SHAREABLE;

    MPU_InitStruct.Number =
        MPU_REGION_NUMBER15;

    MPU_InitStruct.TypeExtField =
        MPU_TEX_LEVEL1;

    MPU_InitStruct.SubRegionDisable = 0x00;

    MPU_InitStruct.DisableExec =
        MPU_INSTRUCTION_ACCESS_ENABLE;

    HAL_MPU_ConfigRegion(&MPU_InitStruct);

    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

    SCB_EnableICache();
    SCB_EnableDCache();
}

/* =========================================================
 * AI initialization
 * ========================================================= */

static int ai_init(void)
{
    if (stai_network_init(network) != 0) {
        LOG_ERR("AI initialization failed");
        return -1;
    }

    if (stai_network_set_activations(
            network,
            act_buffers,
            1
        ) != 0) {
        return -1;
    }

    if (stai_network_set_outputs(
            network,
            out_buffers,
            3
        ) != 0) {
        return -1;
    }

    App_Config.output.pOutBuff = detections;
    App_Config.output.nb_detect = 0;

    app_postprocess_init(&App_Config);

    LOG_INF("AI initialized");

    return 0;
}

/* =========================================================
 * Camera initialization
 * ========================================================= */

static int setup_camera(void)
{
    struct video_format req_fmt{};

    req_fmt.pixelformat = VIDEO_PIX_FMT_RGB565;
    req_fmt.width = CAM_WIDTH;
    req_fmt.height = CAM_HEIGHT;
    req_fmt.pitch = CAM_WIDTH * CAM_BPP;

    if (video_set_format(video_dev, &req_fmt) < 0) {

        if (video_get_format(video_dev, &cam_fmt) < 0) {
            return -1;
        }

    } else {
        cam_fmt = req_fmt;
    }

    if (cam_fmt.pitch == 0) {
        cam_fmt.pitch = cam_fmt.width * CAM_BPP;
    }

    if (cam_fmt.size == 0) {
        cam_fmt.size = cam_fmt.pitch * cam_fmt.height;
    }

    if (cam_fmt.pixelformat != VIDEO_PIX_FMT_RGB565 ||
        cam_fmt.width != CAM_WIDTH ||
        cam_fmt.height != CAM_HEIGHT ||
        cam_fmt.size > MAX_FRAME_SIZE) {

        LOG_ERR("Unexpected camera format");
        return -1;
    }

    LOG_INF(
        "Camera: %ux%u pitch=%u size=%u",
        cam_fmt.width,
        cam_fmt.height,
        cam_fmt.pitch,
        cam_fmt.size
    );

    for (int i = 0; i < CAMERA_BUF_COUNT; i++) {

        struct video_buffer *vb =
            &video_buf_metadata[i];

        vb->buffer = raw_video_data[i];
        vb->size = cam_fmt.size;
        vb->type = VIDEO_BUF_TYPE_OUTPUT;

        if (video_enqueue(video_dev, vb) < 0) {
            return -1;
        }
    }

    return 0;
}

/* =========================================================
 * RGB565 -> RGB888
 * Center crop 192x192
 * ========================================================= */

#pragma GCC push_options
#pragma GCC optimize("O3", "unroll-loops")

static int process_camera_frame(
    const uint8_t *src,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t src_pitch,
    int buf_idx
)
{
    if (src_w < AI_WIDTH || src_h < AI_HEIGHT) {
        return -EINVAL;
    }

    uint32_t crop_x = (src_w - AI_WIDTH) / 2;
    uint32_t crop_y = (src_h - AI_HEIGHT) / 2;

    uint8_t *dst = my_camera_data[buf_idx];

    for (uint32_t y = 0; y < AI_HEIGHT; y++) {

        const uint16_t *src_row =
            (const uint16_t *)(
                src +
                (y + crop_y) * src_pitch +
                crop_x * 2
            );

        for (uint32_t x = 0; x < AI_WIDTH; x++) {

            uint16_t pixel = *src_row++;

            uint8_t r = (pixel >> 11) & 0x1F;
            uint8_t g = (pixel >> 5) & 0x3F;
            uint8_t b = pixel & 0x1F;

            *dst++ = (r << 3) | (r >> 2);
            *dst++ = (g << 2) | (g >> 4);
            *dst++ = (b << 3) | (b >> 2);
        }
    }

    return 0;
}

#pragma GCC pop_options

/* =========================================================
 * JPEG direct output callback
 * ========================================================= */

struct jpeg_output_context {
    uint8_t *dst;

    uint32_t size;
    uint32_t capacity;

    bool overflow;
};

static void jpeg_write_direct(
    void *context,
    void *data,
    int size
)
{
    struct jpeg_output_context *ctx =
        (struct jpeg_output_context *)context;

    if (size <= 0 || ctx->overflow) {
        return;
    }

    uint32_t n = (uint32_t)size;

    if (n > ctx->capacity - ctx->size) {

        ctx->overflow = true;
        return;
    }

    memcpy(
        ctx->dst + ctx->size,
        data,
        n
    );

    ctx->size += n;
}

/* =========================================================
 * TX slot acquisition
 * ========================================================= */

static int acquire_tx_slot(void)
{
    while (1) {

        for (int i = 0; i < TX_SLOT_COUNT; i++) {

            if (k_sem_take(
                    &tx_slot_free[i],
                    K_NO_WAIT
                ) == 0) {

                return i;
            }
        }

        /*
         * Both TX buffers are busy.
         * Wait briefly, then check both slots again.
         */
        k_msleep(1);
    }
}

/* =========================================================
 * Camera thread
 * ========================================================= */

static void camera_thread_entry(
    void *p1,
    void *p2,
    void *p3
)
{
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    int current_buf_idx = 0;

    uint32_t frame_id = 0;

    while (1) {

        struct video_buffer *vbuf = nullptr;

        if (video_dequeue(
                video_dev,
                &vbuf,
                K_MSEC(2000)
            ) < 0) {

            continue;
        }

        if (k_sem_take(
                &buf_free_sem[current_buf_idx],
                K_NO_WAIT
            ) != 0) {

            video_enqueue(video_dev, vbuf);

            dropped_frames++;

            current_buf_idx =
                (current_buf_idx + 1) % PROC_BUF_COUNT;

            continue;
        }

        SCB_InvalidateDCache_by_Addr(
            (uint32_t *)vbuf->buffer,
            cam_fmt.size
        );

        uint32_t start = k_cycle_get_32();

        int ret = process_camera_frame(
            (const uint8_t *)vbuf->buffer,
            cam_fmt.width,
            cam_fmt.height,
            cam_fmt.pitch,
            current_buf_idx
        );

        uint32_t prep_us =
            cycles_to_us(k_cycle_get_32() - start);

        bool queued = false;

        if (ret == 0) {

            struct frame_msg msg{};

            msg.buf_idx = current_buf_idx;
            msg.frame_id = ++frame_id;
            msg.prep_us = prep_us;

            if (k_msgq_put(
                    &frame_msgq,
                    &msg,
                    K_NO_WAIT
                ) == 0) {

                queued = true;
            }
        }

        if (!queued) {

            k_sem_give(&buf_free_sem[current_buf_idx]);

            dropped_frames++;
        }

        video_enqueue(video_dev, vbuf);

        current_buf_idx =
            (current_buf_idx + 1) % PROC_BUF_COUNT;
    }
}

/* =========================================================
 * UART TX thread
 * ========================================================= */

static void uart_tx_thread_entry(
    void *p1,
    void *p2,
    void *p3
)
{
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    while (1) {

        struct tx_msg msg{};

        k_msgq_get(
            &tx_msgq,
            &msg,
            K_FOREVER
        );

        int ret = uart_send_blocking(
            uart_tx_buffer[msg.slot],
            msg.total_len
        );

        if (ret == 0) {

            LOG_INF(
                "[TX] Frame=%u AI=%u us Post=%u us "
                "Prep=%u us JPEG=%u us UART=%u us",
                msg.frame_id,
                msg.ai_us,
                msg.post_us,
                msg.prep_us,
                msg.jpeg_us,
                uart_tx_duration_us
            );

            LOG_INF(
                "[PACKET] Total=%u JPEG=%u Boxes=%u Padding=%u",
                msg.total_len,
                msg.jpeg_size,
                msg.box_count,
                FIXED_PACKET_SIZE -
                    sizeof(frame_header) -
                    msg.box_count * sizeof(bbox_data) -
                    msg.jpeg_size
            );

            LOG_INF(
                "[UART] ISR=%u FIFO=%u MaxGap=%u us "
                "AI bytes=%u AI ISR=%u",
                (unsigned)uart_isr_calls,
                (unsigned)uart_bytes_queued,
                cycles_to_us(uart_max_isr_gap_cycles),
                (unsigned)uart_bytes_during_ai,
                (unsigned)uart_isr_during_ai
            );

        } else {

            LOG_ERR("UART TX failed: %d", ret);
        }

        k_sem_give(&tx_slot_free[msg.slot]);
    }
}

/* =========================================================
 * Main
 * ========================================================= */

int main(void)
{
    Enable_SDRAM_Cache();

    k_msleep(1000);

    LOG_INF("==================================");
    LOG_INF("H747 Edge AI JPEG Optimized V2");
    LOG_INF("Packet Size = %u", FIXED_PACKET_SIZE);
    LOG_INF("UART Baudrate = %u", UART_BAUDRATE);
    LOG_INF("UART = Interrupt / No DMA");
    LOG_INF("JPEG Quality = %u", JPEG_QUALITY);
    LOG_INF("JPEG direct TX buffer");
    LOG_INF("==================================");

    /* UART */

    if (!device_is_ready(uart_dev)) {
        LOG_ERR("UART8 not ready");
        return 0;
    }

    struct uart_config ucfg = {
        .baudrate = UART_BAUDRATE,
        .parity = UART_CFG_PARITY_NONE,
        .stop_bits = UART_CFG_STOP_BITS_1,
        .data_bits = UART_CFG_DATA_BITS_8,
        .flow_ctrl = UART_CFG_FLOW_CTRL_NONE
    };

    if (uart_configure(uart_dev, &ucfg) != 0) {
        LOG_ERR("UART configure failed");
        return 0;
    }

    k_sem_init(&uart_tx_done_sem, 0, 1);

    uart_irq_callback_user_data_set(
        uart_dev,
        uart_isr,
        nullptr
    );

    uart_irq_rx_disable(uart_dev);
    uart_irq_tx_disable(uart_dev);

    /* Semaphores */

    for (int i = 0; i < PROC_BUF_COUNT; i++) {
        k_sem_init(&buf_free_sem[i], 1, 1);
    }

    for (int i = 0; i < TX_SLOT_COUNT; i++) {
        k_sem_init(&tx_slot_free[i], 1, 1);
    }

    /* AI */

    if (ai_init() != 0) {
        LOG_ERR("AI initialization failed");
        return 0;
    }

    /* Camera */

    video_dev = DEVICE_DT_GET_ANY(st_stm32_dcmi);

    if (video_dev == nullptr) {
        video_dev = DEVICE_DT_GET(DT_NODELABEL(ov5640));
    }

    if (video_dev == nullptr ||
        !device_is_ready(video_dev)) {

        LOG_ERR("Camera not ready");
        return 0;
    }

    if (setup_camera() < 0) {
        LOG_ERR("Camera setup failed");
        return 0;
    }

    if (video_stream_start(
            video_dev,
            VIDEO_BUF_TYPE_OUTPUT
        ) < 0) {

        LOG_ERR("Camera stream start failed");
        return 0;
    }

    /* Camera thread */

    k_thread_create(
        &cam_thread_data,
        cam_thread_stack,
        K_THREAD_STACK_SIZEOF(cam_thread_stack),
        camera_thread_entry,
        nullptr,
        nullptr,
        nullptr,
        5,
        0,
        K_NO_WAIT
    );

    /* UART thread */

    k_thread_create(
        &tx_thread_data,
        tx_thread_stack,
        K_THREAD_STACK_SIZEOF(tx_thread_stack),
        uart_tx_thread_entry,
        nullptr,
        nullptr,
        nullptr,
        K_PRIO_COOP(0),
        0,
        K_NO_WAIT
    );

    LOG_INF("System Ready");

    /* =====================================================
     * Main AI / JPEG pipeline
     * ===================================================== */

    while (1) {

        struct frame_msg msg{};

        k_msgq_get(
            &frame_msgq,
            &msg,
            K_FOREVER
        );

        int buf_idx = msg.buf_idx;

        /* AI inference */

        in_buffers[0] = my_camera_data[buf_idx];

        if (stai_network_set_inputs(
                network,
                in_buffers,
                1
            ) != 0) {

            k_sem_give(&buf_free_sem[buf_idx]);
            continue;
        }

        uint32_t ai_start = k_cycle_get_32();

        atomic_set(&ai_running, 1);

        stai_return_code err =
            stai_network_run(network, (stai_run_mode)0);

        atomic_set(&ai_running, 0);

        uint32_t ai_us =
            cycles_to_us(k_cycle_get_32() - ai_start);

        if (err != 0) {

            LOG_ERR("AI inference error: 0x%08X",
                    (unsigned int)err);

            k_sem_give(&buf_free_sem[buf_idx]);
            continue;
        }

        /* Postprocess */

        uint32_t post_start = k_cycle_get_32();

        App_Config.output.nb_detect = 0;

        app_postprocess_run(
            (void **)out_buffers,
            &App_Config.output,
            &App_Config.input_static_param
        );

        uint32_t post_us =
            cycles_to_us(k_cycle_get_32() - post_start);

        /* Bounding boxes */

        struct bbox_data bboxes[MAX_DETECTIONS]{};

        uint16_t valid_boxes = 0;

        for (int i = 0;
             i < App_Config.output.nb_detect &&
             valid_boxes < MAX_DETECTIONS;
             i++) {

            float w = App_Config.output.pOutBuff[i].width;
            float h = App_Config.output.pOutBuff[i].height;

            float cx = App_Config.output.pOutBuff[i].x_center;
            float cy = App_Config.output.pOutBuff[i].y_center;

            float conf = App_Config.output.pOutBuff[i].conf;

            int class_idx =
                App_Config.output.pOutBuff[i].class_index;

            if (class_idx == 0 || conf < 0.60f) {
                continue;
            }

            if (w <= 1.0f && h <= 1.0f) {
                cx *= 192.0f;
                cy *= 192.0f;
                w *= 192.0f;
                h *= 192.0f;
            }

            int x = (int)(cx - w / 2.0f);
            int y = (int)(cy - h / 2.0f);

            if (x < 0) x = 0;
            if (y < 0) y = 0;

            if (x > 191) x = 191;
            if (y > 191) y = 191;

            if (w < 0) w = 0;
            if (h < 0) h = 0;

            if (x + (int)w > 192) w = 192 - x;
            if (y + (int)h > 192) h = 192 - y;

            bboxes[valid_boxes].x = (uint16_t)x;
            bboxes[valid_boxes].y = (uint16_t)y;

            bboxes[valid_boxes].w = (uint16_t)w;
            bboxes[valid_boxes].h = (uint16_t)h;

            bboxes[valid_boxes].conf = conf;

            bboxes[valid_boxes].label_char =
                (class_idx == 1) ? 'C' : '?';

            valid_boxes++;
        }

        /* =================================================
         * Acquire TX slot BEFORE JPEG encoding
         * ================================================= */

        int tx_slot = acquire_tx_slot();

        uint8_t *tx_buffer =
            uart_tx_buffer[tx_slot];

        /*
         * Clear fixed packet.
         * Remaining area is zero padding.
         */

        memset(
            tx_buffer,
            0,
            FIXED_PACKET_SIZE
        );

        size_t boxes_bytes =
            valid_boxes * sizeof(bbox_data);

        size_t payload_offset =
            sizeof(frame_header) + boxes_bytes;

        if (payload_offset >= FIXED_PACKET_SIZE) {

            k_sem_give(&tx_slot_free[tx_slot]);
            k_sem_give(&buf_free_sem[buf_idx]);

            continue;
        }

        /* Copy bounding boxes */

        if (valid_boxes > 0) {

            memcpy(
                tx_buffer + sizeof(frame_header),
                bboxes,
                boxes_bytes
            );
        }

        /* =================================================
         * JPEG direct output
         * ================================================= */

        struct jpeg_output_context jpeg_ctx{};

        jpeg_ctx.dst = tx_buffer + payload_offset;

        jpeg_ctx.capacity =
            FIXED_PACKET_SIZE - payload_offset;

        jpeg_ctx.size = 0;
        jpeg_ctx.overflow = false;

        uint32_t jpeg_start = k_cycle_get_32();

        atomic_set(&jpeg_running, 1);

        int jpeg_ok = stbi_write_jpg_to_func(
            jpeg_write_direct,
            &jpeg_ctx,
            AI_WIDTH,
            AI_HEIGHT,
            AI_CHANNELS,
            my_camera_data[buf_idx],
            JPEG_QUALITY
        );

        atomic_set(&jpeg_running, 0);

        uint32_t jpeg_us =
            cycles_to_us(k_cycle_get_32() - jpeg_start);

        /* Release AI buffer after JPEG finishes */

        k_sem_give(&buf_free_sem[buf_idx]);

        /* JPEG validation */

        if (!jpeg_ok ||
            jpeg_ctx.overflow ||
            jpeg_ctx.size < 4 ||
            jpeg_ctx.size > MAX_JPEG_SIZE) {

            LOG_ERR(
                "JPEG invalid: size=%u overflow=%d",
                jpeg_ctx.size,
                (int)jpeg_ctx.overflow
            );

            k_sem_give(&tx_slot_free[tx_slot]);
            dropped_frames++;

            continue;
        }

        uint8_t *jpeg_data =
            tx_buffer + payload_offset;

        if (jpeg_data[0] != 0xFF ||
            jpeg_data[1] != 0xD8 ||
            jpeg_data[jpeg_ctx.size - 2] != 0xFF ||
            jpeg_data[jpeg_ctx.size - 1] != 0xD9) {

            LOG_ERR("JPEG marker invalid");

            k_sem_give(&tx_slot_free[tx_slot]);

            continue;
        }

        /* =================================================
         * Build packet header
         * ================================================= */

        frame_header hdr{};

        hdr.magic_word = 0xAABB;
        hdr.image_size = jpeg_ctx.size;
        hdr.box_count = valid_boxes;

        memcpy(
            tx_buffer,
            &hdr,
            sizeof(hdr)
        );

        /*
         * Packet length is always 6000 bytes.
         */

        struct tx_msg tx{};

        tx.slot = (uint8_t)tx_slot;

        tx.frame_id = msg.frame_id;

        tx.total_len = FIXED_PACKET_SIZE;

        tx.prep_us = msg.prep_us;
        tx.ai_us = ai_us;
        tx.post_us = post_us;
        tx.jpeg_us = jpeg_us;

        tx.jpeg_size = jpeg_ctx.size;
        tx.box_count = valid_boxes;

        /* =================================================
         * JPEG statistics
         * ================================================= */

        if (jpeg_us < jpeg_min_us) {
            jpeg_min_us = jpeg_us;
        }

        if (jpeg_us > jpeg_max_us) {
            jpeg_max_us = jpeg_us;
        }

        jpeg_total_us += jpeg_us;
        jpeg_count++;

        frame_count++;

        /* =================================================
         * Queue UART transmission
         * ================================================= */

        if (k_msgq_put(
                &tx_msgq,
                &tx,
                K_FOREVER
            ) != 0) {

            k_sem_give(&tx_slot_free[tx_slot]);
            continue;
        }

        /* =================================================
         * Periodic statistics
         * ================================================= */

        if ((frame_count % 10) == 0) {

            uint32_t avg_jpeg =
                (uint32_t)(jpeg_total_us / jpeg_count);

            LOG_INF(
                "[JPEG STATS] Min=%u us Avg=%u us Max=%u us",
                jpeg_min_us,
                avg_jpeg,
                jpeg_max_us
            );

            LOG_INF(
                "[SYSTEM] Frames=%u Dropped=%u",
                frame_count,
                dropped_frames
            );
        }
    }

    return 0;
}
