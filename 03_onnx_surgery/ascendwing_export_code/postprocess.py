import cv2
import numpy as np


def decode(bbox_preds, cls_scores, top_k=300):
    """CPU-side NMS-free decoder.

    bbox_preds: [8400, 4]  -- decoded xyxy boxes in 640x640 pixel space
    cls_scores: [8400, N]  -- class confidence scores (after sigmoid)
    Returns: [top_k, 6] = [x1, y1, x2, y2, conf, cls]
    """
    max_scores = cls_scores.max(axis=1)
    max_classes = cls_scores.argmax(axis=1)

    indices = np.argpartition(-max_scores, top_k - 1)[:top_k]
    scores = max_scores[indices]
    order = np.argsort(-scores)
    indices = indices[order]
    scores = scores[order]
    classes = max_classes[indices]
    boxes = bbox_preds[indices]

    return np.column_stack([boxes, scores[:, None], classes[:, None]])


def scale_boxes(boxes, ratio, dw, dh, orig_shape):
    """Scale boxes from 640x640 letterbox space back to original image."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def filter_by_conf(dets, conf_thres):
    return dets[dets[:, 4] > conf_thres]


def draw(image, dets, labels, conf_thres=0.25, color=(0, 255, 0)):
    """Draw detection boxes on image (in-place)."""
    for d in dets:
        x1, y1, x2, y2, conf, cls = d
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{labels.get(int(cls), str(int(cls)))} {conf:.2f}'
        cv2.putText(image, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
