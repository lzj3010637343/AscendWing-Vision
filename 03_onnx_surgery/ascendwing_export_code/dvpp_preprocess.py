"""
DVPP hardware-accelerated preprocessing for YOLO26 on Ascend 310B1.

Pipeline: JPEG bytes → DVPP decode (YUV) → host copy → feed to AIPP(YUV) model

AIPP handles: YUV→RGB + resize + normalize, all in hardware.
Replaces: cv2.imread + cv2.resize + transpose + /255.

Usage:
    from dvpp_preprocess import DvppPreprocessor

    proc = DvppPreprocessor()
    yuv_buffer, img_w, img_h = proc.decode('image.jpg')
    model.infer([yuv_buffer])  # feeds YUV420SP directly to AIPP model
"""
import numpy as np
import acl
import os
import atexit

PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1  # PIXEL_FORMAT_YUV_SEMIPLANAR_420
ACL_SUCCESS = 0


def align_up(x, n):
    return ((x + n - 1) // n) * n


class DvppPreprocessor:
    def __init__(self):
        self._dvpp_channel_desc = None
        self._stream = None
        self._init_dvpp()
        atexit.register(self.destroy)

    def _init_dvpp(self):
        # Create stream
        self._stream, ret = acl.rt.create_stream()
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.create_stream failed: {ret}")

        # Create DVPP channel
        self._dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self._dvpp_channel_desc)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.media.dvpp_create_channel failed: {ret}")

    def decode(self, jpeg_path_or_bytes):
        """DVPP JPEG decode → YUV420SP host buffer.
        Returns (numpy_array, width, height).
        """
        if isinstance(jpeg_path_or_bytes, str):
            data = np.fromfile(jpeg_path_or_bytes, dtype=np.uint8)
        else:
            data = np.asarray(jpeg_path_or_bytes, dtype=np.uint8)

        data_ptr = acl.util.bytes_to_ptr(data.tobytes())
        data_size = data.nbytes

        # Get image info (width, height)
        img_info = acl.media.dvpp_jpeg_get_image_info_v2(data_ptr, data_size)
        ret = img_info[-1] if isinstance(img_info, tuple) else img_info
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"dvpp_jpeg_get_image_info_v2 failed: {ret}")
        img_w, img_h = img_info[0], img_info[1]

        # Predict decode buffer size
        dec_size, ret = acl.media.dvpp_jpeg_predict_dec_size(
            data_ptr, data_size, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"dvpp_jpeg_predict_dec_size failed: {ret}")

        # Allocate device output buffer
        out_buffer, ret = acl.media.dvpp_malloc(dec_size)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"dvpp_malloc failed: {ret}")

        # Create output pic desc
        out_desc = acl.media.dvpp_create_pic_desc()
        stride_w = align_up(img_w, 128)  # JPEG decode requires 128 alignment
        stride_h = align_up(img_h, 16)

        acl.media.dvpp_set_pic_desc_data(out_desc, out_buffer)
        acl.media.dvpp_set_pic_desc_format(out_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(out_desc, img_w)
        acl.media.dvpp_set_pic_desc_height(out_desc, img_h)
        acl.media.dvpp_set_pic_desc_width_stride(out_desc, stride_w)
        acl.media.dvpp_set_pic_desc_height_stride(out_desc, stride_h)
        acl.media.dvpp_set_pic_desc_size(out_desc, dec_size)

        # JPEG decode async
        ret = acl.media.dvpp_jpeg_decode_async(
            self._dvpp_channel_desc, data_ptr, data_size, out_desc, self._stream)
        if ret != ACL_SUCCESS:
            acl.media.dvpp_free(out_buffer)
            acl.media.dvpp_destroy_pic_desc(out_desc)
            raise RuntimeError(f"dvpp_jpeg_decode_async failed: {ret}")

        ret = acl.rt.synchronize_stream(self._stream)
        if ret != ACL_SUCCESS:
            acl.media.dvpp_free(out_buffer)
            acl.media.dvpp_destroy_pic_desc(out_desc)
            raise RuntimeError(f"synchronize_stream failed: {ret}")

        # Copy YUV from device to host
        # YUV420SP size = W*H*3/2 (Y plane + interleaved UV plane)
        yuv_size = img_w * img_h * 3 // 2
        host_buf = np.zeros(yuv_size, dtype=np.uint8)
        host_ptr = acl.util.bytes_to_ptr(host_buf.tobytes())
        ret = acl.rt.memcpy(host_ptr, yuv_size, out_buffer, yuv_size,
                            acl.constants.ACL_MEMCPY_DEVICE_TO_HOST)
        if ret != ACL_SUCCESS:
            acl.media.dvpp_free(out_buffer)
            acl.media.dvpp_destroy_pic_desc(out_desc)
            raise RuntimeError(f"memcpy device->host failed: {ret}")

        # Cleanup
        acl.media.dvpp_free(out_buffer)
        acl.media.dvpp_destroy_pic_desc(out_desc)

        return host_buf, img_w, img_h

    def destroy(self):
        if self._dvpp_channel_desc:
            acl.media.dvpp_destroy_channel(self._dvpp_channel_desc)
            acl.media.dvpp_destroy_channel_desc(self._dvpp_channel_desc)
            self._dvpp_channel_desc = None
        if self._stream:
            acl.rt.destroy_stream(self._stream)
            self._stream = None


def gen_aipp_cfg_yuv(src_w, src_h, dst_w=640, dst_h=640):
    """Generate AIPP config for YUV420SP input with CSC + resize + normalize."""
    # BT.601 YUV->RGB, fixed-point ×256
    return f"""aipp_op {{
    aipp_mode: static
    related_input_rank: 0
    input_format: YUV420SP_U8
    src_image_size_w: {src_w}
    src_image_size_h: {src_h}
    crop: false
    load_start_pos_w: 0
    load_start_pos_h: 0
    csc_switch: true
    rbuv_swap_switch: false
    matrix_r0c0: 298
    matrix_r0c1: 0
    matrix_r0c2: 409
    matrix_r1c0: 298
    matrix_r1c1: -100
    matrix_r1c2: -208
    matrix_r2c0: 298
    matrix_r2c1: 517
    matrix_r2c2: 0
    input_bias_0: 16
    input_bias_1: 128
    input_bias_2: 128
    var_reci_chn_0: 0.00392157
    var_reci_chn_1: 0.00392157
    var_reci_chn_2: 0.00392157
}}
"""
