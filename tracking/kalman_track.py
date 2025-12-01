# tracking/kalman_track.py

import time
import numpy as np
import config
from reid.reid_helper import cosine_similarity  # Se asume que existe


class KalmanTrack:
    def __init__(self, track_id, init_bbox, init_emb, dt):
        x1, y1, x2, y2 = init_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.dt = dt

        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0

        # Matrices Kalman
        q = 1.0
        r = 5.0
        self.Q = np.diag([q, q, q * 0.5, q * 0.5]).astype(np.float32)
        self.R = np.diag([r, r]).astype(np.float32)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.id = track_id
        self.last_seen = time.time()
        self.age = 0
        self.consecutive_missed = 0
        self.w = float(x2 - x1)
        self.h = float(y2 - y1)
        self.ema_alpha = 0.6

        self.prototype = init_emb.copy() if init_emb is not None else None
        self.bbox = init_bbox

    def predict(self):
        # --- CAMBIO: Si ya está perdido, no muevas nada ---
        if self.consecutive_missed > 0:
            # Solo aumentamos la incertidumbre (P) y la edad, pero NO movemos X
            # Esto evita que la caja "derive" por ruido matemático.
            self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
            self.age += 1
            return self.x[0], self.x[1]
        # --------------------------------------------------
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.age += 1
        return self.x[0], self.x[1]

    def update(self, bbox, emb):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        z = np.array([cx, cy], dtype=np.float32)

        # Fusión Kalman
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K.dot(self.H)).dot(self.P)

        # Actualizar tamaño (EMA)
        w_new = float(x2 - x1)
        h_new = float(y2 - y1)
        self.w = self.ema_alpha * self.w + (1 - self.ema_alpha) * w_new
        self.h = self.ema_alpha * self.h + (1 - self.ema_alpha) * h_new
        self.bbox = bbox

        # Actualizar ReID prototype (EMA)
        if emb is not None:
            if self.prototype is None:
                self.prototype = emb.copy()
            else:
                m = config.EMBED_UPDATE_MOMENTUM
                self.prototype = m * self.prototype + (1 - m) * emb
                self.prototype = self.prototype / (np.linalg.norm(self.prototype) + 1e-6)

        self.last_seen = time.time()
        self.consecutive_missed = 0

    def mark_missed(self):
        """
        Incrementa el contador de frames perdidos y FRENA la velocidad
        para evitar que el fantasma salga volando.
        """
        self.consecutive_missed += 1
        # --- AGREGA ESTO: FRENADO DE EMERGENCIA ---
        # Multiplicamos la velocidad (índices 2 y 3 del estado) por un factor bajo (ej. 0.1)
        # Esto hace que la caja deje de predecir movimiento si no ve al cerdo.
        self.x[2] = 0 # vx
        self.x[3] *= 0 # vy
        # ------------------------------------------

    def get_predicted_bbox(self):
        cx, cy = self.x[0], self.x[1]
        w, h = self.w, self.h
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (int(x1), int(y1), int(x2), int(y2))

    def motion_distance(self, bbox, img_diag):
        px, py = self.x[0], self.x[1]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        return d / (img_diag + 1e-6)