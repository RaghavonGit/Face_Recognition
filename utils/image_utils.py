import cv2
import numpy as np
import os

def imread_unicode(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def imwrite_unicode(path, img):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)
        return True
    except: return False