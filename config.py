import os
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))

# --- PATHS ---
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
SCRFD_10G_ONNX = os.path.join(CHECKPOINT_DIR, "scrfd_10g_bnkps.onnx")
CHECKPOINT_TEACHER = os.path.join(CHECKPOINT_DIR, "teacher_resnet100.pth")

RAW_ROOT = os.path.join(ROOT, "dataset", "raw")
ALIGNED_ROOT = os.path.join(ROOT, "dataset", "aligned")

DB_PATH = os.path.join(ROOT, "attendance.db")
FAISS_INDEX_PATH = os.path.join(ROOT, "vectors.index")
ID_MAP_PATH = os.path.join(ROOT, "id_map.pkl")
FERNET_KEY_PATH = os.path.join(ROOT, "fernet.key")

# --- SETTINGS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection
DET_INPUT_SIZE = (640, 640)
STRIDES = [8, 16, 32]
CONF_THRESHOLD = 0.5        # Ignore weak detections
NMS_IOU = 0.45

# Alignment
ALIGN_SIZE = (112, 112)
MIN_FACE_SIZE = (60, 60)    # Ignore small background noise

# Recognition
EMBEDDING_SIZE = 512
REC_THRESHOLD = 0.65        # Strictness (0.65 filters out cousins)