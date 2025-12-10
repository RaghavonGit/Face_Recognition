import cv2
import numpy as np
import config
from detector_scrfd import AutoSCRFD
from align_face import align_face
from recognizer import Recognizer
from enrollment_system import EnrollmentSystem

det = AutoSCRFD(config.SCRFD_10G_ONNX)
rec = Recognizer()
enr = EnrollmentSystem()
cap = cv2.VideoCapture(0)
stack = []

print("PRO SYSTEM | [E] Add Angle | [S] Save | [C] Clear | [Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    dets = det.detect(frame)
    
    # Filter small/multiple faces
    valid = [d for d in dets if (d['bbox'][2]-d['bbox'][0]) > config.MIN_FACE_SIZE[0]]
    if valid:
        # Pick largest
        d = max(valid, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
        x1, y1, x2, y2 = map(int, d["bbox"])
        
        # Blur check
        if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 60:
            color, txt = (0, 255, 255), "Blurry"
        else:
            aligned = align_face(frame, d["kps"])
            if aligned is not None:
                emb = rec.get_embedding(aligned)
                name, conf = enr.search(emb)
                
                if stack: color, txt = (255, 0, 0), f"Stack: {len(stack)}"
                elif name != "Unknown": color, txt = (0, 255, 0), f"{name} ({conf:.2f})"
                else: color, txt = (0, 0, 255), "Unknown"
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord('e'): stack.append(emb); print(f"Added. Total: {len(stack)}")
                elif k == ord('s') and stack:
                    avg = np.mean(stack, axis=0); avg /= np.linalg.norm(avg)
                    enr.add_user(input("Name: "), avg); stack = []
                elif k == ord('c'): stack = []

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, y1-10), 0, 0.8, color, 2)

    cv2.imshow("Face System", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()