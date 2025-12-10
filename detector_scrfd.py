import onnxruntime as ort
import numpy as np
import cv2
from config import DET_INPUT_SIZE, STRIDES, CONF_THRESHOLD, NMS_IOU

INPUT_SIZE = DET_INPUT_SIZE

def generate_centers(w, h, stride):
    fw = int(np.ceil(w / stride))
    fh = int(np.ceil(h / stride))
    cx = np.arange(fw) * stride + stride / 2
    cy = np.arange(fh) * stride + stride / 2
    cxg, cyg = np.meshgrid(cx, cy)
    return np.stack([cxg.reshape(-1), cyg.reshape(-1)], axis=-1).astype(np.float32)

def decode_bboxes(centers, preds):
    x1 = centers[:, 0] - preds[:, 0]
    y1 = centers[:, 1] - preds[:, 1]
    x2 = centers[:, 0] + preds[:, 2]
    y2 = centers[:, 1] + preds[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def decode_kps(centers, preds):
    return preds.reshape(-1, 5, 2) + centers[:, None, :]

def nms(boxes, scores, thr):
    if boxes.shape[0] == 0: return []
    x1, y1, x2, y2 = boxes.T
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / ((x2[i]-x1[i])*(y2[i]-y1[i]) + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]]) - inter + 1e-6)
        inds = np.where(iou <= thr)[0]
        order = order[inds + 1]
    return keep

class OnnxScrfdDetector:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = INPUT_SIZE
        self.priors_list = [generate_centers(self.input_size[0], self.input_size[1], s) for s in STRIDES]

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def detect(self, img_bgr, conf_thresh=CONF_THRESHOLD):
        if img_bgr is None: return []
        canvas, scale = self.preprocess(img_bgr)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5
        blob /= 128.0
        blob = blob.transpose(2, 0, 1)[None]
        
        outputs = self.session.run(None, {self.input_name: blob})
        scores_list, bboxes_list, kps_list = [], [], []
        
        for idx, stride in enumerate(STRIDES):
            score_pred = outputs[idx].reshape(-1)
            box_pred = outputs[idx+3] * stride
            kps_pred = outputs[idx+6] * stride
            priors = self.priors_list[idx]
            
            if len(score_pred) > len(priors):
                priors = np.repeat(priors, len(score_pred)//len(priors), axis=0)
            
            scores_list.append(score_pred)
            bboxes_list.append(decode_bboxes(priors, box_pred))
            kps_list.append(decode_kps(priors, kps_pred))

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        kps = np.concatenate(kps_list, axis=0)

        keep = np.where(scores > conf_thresh)[0]
        bboxes, kps, scores = bboxes[keep], kps[keep], scores[keep]
        
        keep_nms = nms(bboxes, scores, NMS_IOU)
        
        results = []
        for i in keep_nms:
            b = bboxes[i] / scale
            k = kps[i] / scale
            results.append({"bbox": b.tolist(), "kps": k.tolist(), "score": float(scores[i])})
        return results

class AutoSCRFD:
    def __init__(self, preferred, fallback=None):
        self.det = OnnxScrfdDetector(preferred)
    def detect(self, img):
        return self.det.detect(img)