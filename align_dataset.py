import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from utils.image_utils import imread_unicode, imwrite_unicode
from detector_scrfd import AutoSCRFD
from align_face import align_face
import config

# Global worker variable
PREF = None

def init_worker(model_path):
    """Initialize the detector once per worker process"""
    global PREF
    # Load 10G model only
    PREF = AutoSCRFD(model_path, None)

def process_item(task):
    """Process a single image"""
    src_path, dst_path = task
    
    # 1. Read Image safely
    img = imread_unicode(src_path)
    if img is None:
        return (src_path, False, "Read Error")
    
    # 2. Detect Faces
    try:
        dets = PREF.detect(img)
    except Exception as e:
        return (src_path, False, f"Detector Crash: {e}")

    if not dets:
        return (src_path, False, "No Face Found")
    
    # 3. Filter: Remove small noise/background spots
    valid_dets = []
    for d in dets:
        w = d['bbox'][2] - d['bbox'][0]
        h = d['bbox'][3] - d['bbox'][1]
        
        # Must be larger than config setting (e.g. 60px)
        if w > config.MIN_FACE_SIZE[0] and h > config.MIN_FACE_SIZE[1]:
            valid_dets.append(d)
            
    if not valid_dets:
        return (src_path, False, "Face Too Small")

    # 4. Pick the Largest Face
    det = max(valid_dets, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
    
    # 5. Align
    aligned = align_face(img, det["kps"])
    
    if aligned is None:
        return (src_path, False, "Alignment Math Failed")
    
    # 6. Save
    ok = imwrite_unicode(dst_path, aligned)
    return (src_path, ok, "Success" if ok else "Write Error")

def collect_pairs(src_root, dst_root, limit=None):
    pairs = []
    count = 0
    print("Scanning dataset files...")
    
    for root, _, files in os.walk(src_root):
        files = sorted(files)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src = os.path.join(root, f)
                
                # Mirror folder structure in destination
                rel_path = os.path.relpath(src, src_root)
                dst = os.path.join(dst_root, rel_path)
                
                # Ensure folder exists
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # Skip if already done
                if not os.path.exists(dst):
                    pairs.append((src, dst))
                    count += 1
                    if limit and count >= limit:
                        return pairs
    return pairs

def align_and_save(src, dst, preferred, workers, limit):
    pairs = collect_pairs(src, dst, limit)
    
    if not pairs:
        print("No new images found to process.")
        return

    print(f"Processing {len(pairs)} images...")
    print(f" - Min Face Size: {config.MIN_FACE_SIZE}")
    print(f" - Threshold: {config.CONF_THRESHOLD}")
    
    results = []
    succ = 0
    
    # Start Progress Bar
    pbar = tqdm(total=len(pairs))

    # Multiprocessing Pool
    if workers > 1:
        with Pool(processes=workers, initializer=init_worker, initargs=(preferred,)) as p:
            for r in p.imap_unordered(process_item, pairs):
                results.append(r)
                if r[1]: succ += 1
                pbar.set_postfix(ok=succ)
                pbar.update(1)
    else:
        # Single Thread (for debugging)
        init_worker(preferred)
        for pair in pairs:
            r = process_item(pair)
            results.append(r)
            if r[1]: succ += 1
            pbar.set_postfix(ok=succ)
            pbar.update(1)

    pbar.close()
    
    # Write Log
    log_path = os.path.join(dst, "align_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for src, ok, reason in results:
            status = "OK" if ok else "FAIL"
            f.write(f"[{status}] {reason} : {src}\n")
                
    print(f"Done. Aligned: {succ} / {len(pairs)}")
    print(f"Check {log_path} for details.")

if __name__ == "__main__":
    # Ensure aligned dir exists
    if not os.path.exists(config.ALIGNED_ROOT):
        os.makedirs(config.ALIGNED_ROOT)

    align_and_save(
        src=config.RAW_ROOT, 
        dst=config.ALIGNED_ROOT, 
        preferred=config.SCRFD_10G_ONNX, 
        workers=4, 
        limit=None
    )