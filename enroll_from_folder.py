import os
import numpy as np
from recognizer import Recognizer
from enrollment_system import EnrollmentSystem
from utils.image_utils import imread_unicode
import config

def enroll_all():
    """
    Scans the 'dataset/aligned' folder and enrolls every person found 
    into the database.
    """
    rec = Recognizer()
    enr = EnrollmentSystem()
    
    aligned_root = config.ALIGNED_ROOT # dataset/aligned
    
    if not os.path.exists(aligned_root):
        print(f"Error: Folder not found at {aligned_root}")
        return

    print(f"--- STARTING ENROLLMENT ---")
    
    count = 0
    # Loop through every person folder (e.g. dataset/aligned/Raghav)
    for person_name in os.listdir(aligned_root):
        person_dir = os.path.join(aligned_root, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        print(f" -> Processing: {person_name}")
        
        # Get all images for this person
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            print("    (No images found)")
            continue
            
        embeddings = []
        
        # Convert every image to numbers (embedding)
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = imread_unicode(img_path)
            
            if img is not None:
                try:
                    # Get the 512-D vector
                    emb = rec.get_embedding(img)
                    embeddings.append(emb)
                except Exception as e:
                    pass # Skip bad images
        
        # Average the vectors to create one "Master Face"
        if embeddings:
            final_emb = np.mean(embeddings, axis=0)
            final_emb = final_emb / np.linalg.norm(final_emb)
            
            uid = enr.add_user(person_name, final_emb)
            print(f"    Success! Enrolled ID: {uid}")
            count += 1
        else:
            print("    Failed (No valid embeddings).")

    print(f"Done. Total Enrolled: {count}")

if __name__ == "__main__":
    enroll_all()