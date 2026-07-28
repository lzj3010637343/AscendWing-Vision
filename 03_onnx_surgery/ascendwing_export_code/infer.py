"""
YOLO26-N on Ascend 310B1 (Atlas 200I DK A2).

Pipeline:
  NPU  -> backbone + neck + detection head -> bbox_preds + cls_scores
  CPU  -> TopK decode -> scale boxes -> draw

Usage:
  python infer.py image  <path>            # single image
  python infer.py video  <path>            # video file
  python infer.py camera [camera_id]       # live camera (default 0)
"""
import sys
import time
import cv2
import numpy as np
from ais_bench.infer.interface import InferSession

from preprocess import preprocess
from postprocess import decode, scale_boxes, filter_by_conf, draw

MODEL_PATH = '/root/yolo26/yolo26n_aipp.om'
LABEL_PATH = '/root/yolo26/labels.txt'
CONF_THRES = 0.25
INPUT_SHAPE = (640, 640)
USE_AIPP = True  # AIPP handles BGR→RGB + /255 in hardware


def load_labels(path):
    with open(path) as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def infer_image(model, image, labels):
    img, ratio, dw, dh = preprocess(image, use_aipp=USE_AIPP)
    raw = model.infer([img])
    dets = decode(raw[0][0], raw[1][0])
    dets = filter_by_conf(dets, CONF_THRES)
    if len(dets):
        dets[:, :4] = scale_boxes(dets[:, :4], ratio, dw, dh, image.shape)
    return dets


def run_image(model, labels, path):
    image = cv2.imread(path)
    if image is None:
        print(f'Cannot read: {path}')
        return

    t0 = time.time()
    dets = infer_image(model, image, labels)
    elapsed = (time.time() - t0) * 1000

    print(f'{len(dets)} detections ({elapsed:.1f}ms)')
    for d in dets[:10]:
        x1, y1, x2, y2, conf, cls = d
        print(f'  {labels.get(int(cls), int(cls)):15s} {conf:.3f}  '
              f'[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]')

    out = draw(image, dets, labels, CONF_THRES)
    cv2.imwrite('output.jpg', out)
    print('Saved output.jpg')


def run_video(model, labels, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'Cannot open: {path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video: {path} ({fps:.1f} fps). Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = infer_image(model, frame, labels)
        draw(frame, dets, labels, CONF_THRES)

        cv2.imshow('YOLO26', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_camera(model, labels, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        # Try other indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cam_id = i
                break
        else:
            print('No camera found')
            return

    print(f'Camera {cam_id}. Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = infer_image(model, frame, labels)
        draw(frame, dets, labels, CONF_THRES)

        cv2.imshow('YOLO26', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    labels = load_labels(LABEL_PATH)
    model = InferSession(0, MODEL_PATH)

    mode = sys.argv[1].lower()

    if mode == 'image':
        path = sys.argv[2] if len(sys.argv) > 2 else 'test.jpg'
        run_image(model, labels, path)

    elif mode == 'video':
        path = sys.argv[2] if len(sys.argv) > 2 else 'test.mp4'
        run_video(model, labels, path)

    elif mode == 'camera':
        cam_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_camera(model, labels, cam_id)

    else:
        print(f'Unknown mode: {mode}')
        print(__doc__)


if __name__ == '__main__':
    main()
