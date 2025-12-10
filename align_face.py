import cv2
import numpy as np
from config import ALIGN_SIZE

# Standard 5 landmarks for 112x112 image
# [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
REFERENCE_PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, kps):
    """
    img: The full input image
    kps: The 5 facial landmarks from the detector
    """
    try:
        dst = REFERENCE_PTS
        src = np.array(kps, dtype=np.float32)

        # Calculate the transformation matrix (Similarity Transform)
        # This handles Rotation, Scale, and Translation
        tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        
        # If the math fails (e.g. points are collinear), return None
        if tform is None:
            return None
            
        # Apply the warp
        warped = cv2.warpAffine(img, tform, ALIGN_SIZE, borderValue=0.0)
        return warped
        
    except Exception as e:
        # In case of weird math errors
        return None