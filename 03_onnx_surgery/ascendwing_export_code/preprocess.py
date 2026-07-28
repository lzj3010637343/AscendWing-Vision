import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to new_shape, keeping aspect ratio."""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(image, use_aipp=True):
    """
    Preprocess BGR image for YOLO26-N NPU inference.

    With AIPP: BGR uint8 HWC → AIPP handles BGR→RGB + normalize + HWC→CHW
    Without:   BGR float32 CHW, /255 normalized
    """
    img, ratio, (dw, dh) = letterbox(image)
    if use_aipp:
        img = np.ascontiguousarray(img)
    else:
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
    return img, ratio, dw, dh
