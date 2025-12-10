import os
import sys
import cv2
import numpy as np
import config

# Import logic from your existing modules
from align_dataset import align_and_save
from enroll_from_folder import enroll_all
from detector_scrfd import AutoSCRFD
from align_face import align_face
from recognizer import Recognizer
from enrollment_system import EnrollmentSystem

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def step_1_process_data():
    print("\n--- STEP 1: PROCESSING DATASET ---")
    print(f"Reading from: {config.RAW_ROOT}")
    print(f"Saving to:   {config.ALIGNED_ROOT}")
    
    # Run the alignment logic using 10G model
    align_and_save(
        src=config.RAW_ROOT, 
        dst=config.ALIGNED_ROOT, 
        preferred=config.SCRFD_10G_ONNX, 
        workers=4, 
        limit=None
    )
    print("\n[Done] Alignment complete.")
    
    print("\n--- STEP 2: UPDATING DATABASE ---")
    # Run enrollment logic
    enroll_all()
    print("\n[Done] Database updated.")
    
    input("\nPress Enter to return to menu...")

def step_2_live_system():
    print("\n--- STARTING LIVE RECOGNITION ---")
    print("Controls:")
    print(" [E] Add Angle (Stacking)")
    print(" [S] Save Stack")
    print(" [C] Clear Stack")
    print(" [Q] Quit Camera")
    
    # Initialize Models
    try:
        det = AutoSCRFD(config.SCRFD_10G_ONNX)
        rec = Recognizer()
        enr = EnrollmentSystem()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        input("Press Enter to exit...")
        return

    cap = cv2.VideoCapture(0)
    stack = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Blur Check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_blurry = cv2.Laplacian(gray, cv2.CV_64F).var() < 60

        # 2. Detect
        dets = det.detect(frame)
        
        # 3. Filter & Process
        valid = [d for d in dets if (d['bbox'][2]-d['bbox'][0]) > config.MIN_FACE_SIZE[0]]
        
        if valid:
            # Pick largest face
            d = max(valid, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
            x1, y1, x2, y2 = map(int, d["bbox"])
            
            if is_blurry:
                color, txt = (0, 255, 255), "Blurry"
            else:
                aligned = align_face(frame, d["kps"])
                if aligned is not None:
                    emb = rec.get_embedding(aligned)
                    name, conf = enr.search(emb)
                    
                    # Logic for UI colors
                    if stack: 
                        color = (255, 0, 0) # Blue for Stacking
                        txt = f"Stack: {len(stack)}"
                    elif name != "Unknown": 
                        color = (0, 255, 0) # Green for Known
                        txt = f"{name} ({conf:.2f})"
                    else: 
                        color = (0, 0, 255) # Red for Unknown
                        txt = "Unknown"
                    
                    # Key Inputs
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('e'): 
                        stack.append(emb)
                        print(f" -> Added angle {len(stack)}")
                    elif k == ord('c'): 
                        stack = []
                        print(" -> Stack cleared")
                    elif k == ord('s') and stack:
                        print(f" -> Saving {len(stack)} angles...")
                        avg = np.mean(stack, axis=0)
                        avg /= np.linalg.norm(avg)
                        
                        # Pause video to get input safely
                        user_name = input(">> Enter Name for new user: ")
                        enr.add_user(user_name, avg)
                        print(f" -> Success! Enrolled {user_name}")
                        stack = []

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live System (Press Q to Quit)", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        clear_screen()
        print("=======================================")
        print("   FACE RECOGNITION COMMAND CENTER")
        print("=======================================")
        print(" 1. Process New Photos (Align & Enroll)")
        print(" 2. Start Live System")
        print(" 3. Exit")
        print("=======================================")
        
        choice = input("Select Option (1-3): ").strip()
        
        if choice == '1':
            step_1_process_data()
        elif choice == '2':
            step_2_live_system()
        elif choice == '3':
            print("Exiting...")
            sys.exit()
        else:
            input("Invalid option. Press Enter...")

if __name__ == "__main__":
    # Required for Windows multiprocessing support
    from multiprocessing import freeze_support
    freeze_support()
    main()