import torch
import cv2
import numpy as np
import os
from models.iresnet import IResNet100
import config

class Recognizer:
    def __init__(self):
        self.device = config.DEVICE
        self.model = IResNet100(config.EMBEDDING_SIZE)
        
        if os.path.exists(config.CHECKPOINT_TEACHER):
            self.model.load_state_dict(torch.load(config.CHECKPOINT_TEACHER, map_location=self.device))
        else:
            print("ERROR: Teacher weights missing!")
            
        self.model.to(self.device).eval()

    def get_embedding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            emb = self.model(t)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten()