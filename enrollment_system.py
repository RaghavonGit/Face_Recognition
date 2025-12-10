import sqlite3
import pickle
import faiss
import os
import numpy as np
from cryptography.fernet import Fernet
import config

class EnrollmentSystem:
    def __init__(self):
        self.conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)")
        self.conn.commit()

        self.index = faiss.IndexFlatIP(config.EMBEDDING_SIZE)
        self.id_map = {}
        
        if os.path.exists(config.FAISS_INDEX_PATH):
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
            with open(config.ID_MAP_PATH, "rb") as f: self.id_map = pickle.load(f)

        if not os.path.exists(config.FERNET_KEY_PATH):
            open(config.FERNET_KEY_PATH, "wb").write(Fernet.generate_key())
        self.cipher = Fernet(open(config.FERNET_KEY_PATH, "rb").read())

    def add_user(self, name, emb):
        faiss.normalize_L2(emb.reshape(1, -1))
        self.cur.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, self.cipher.encrypt(emb.tobytes())))
        uid = self.cur.lastrowid
        self.conn.commit()
        self.index.add(emb.reshape(1, -1))
        self.id_map[self.index.ntotal - 1] = uid
        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        with open(config.ID_MAP_PATH, "wb") as f: pickle.dump(self.id_map, f)
        return uid

    def search(self, emb):
        faiss.normalize_L2(emb.reshape(1, -1))
        if self.index.ntotal == 0: return ("Unknown", 0.0)
        D, I = self.index.search(emb.reshape(1, -1), 1)
        score = float(D[0][0])
        
        if score < config.REC_THRESHOLD: return ("Unknown", score)
        
        uid = self.id_map.get(int(I[0][0]))
        if not uid: return ("Unknown", score)
        
        self.cur.execute("SELECT name FROM users WHERE id=?", (uid,))
        return (self.cur.fetchone()[0], score)