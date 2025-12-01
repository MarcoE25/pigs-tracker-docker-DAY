# tracking/tracker_manager.py
import numpy as np
import time
from tracking.kalman_track import KalmanTrack
from reid.reid_helper import cosine_similarity
def center_of(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea + 1e-6
    return interArea / union
# -------------------------------------------
class TrackerManager:
    """Gestiona múltiples rastreadores Kalman."""
    def __init__(self, dt, max_missed_frames):
        self.tracks = {}  # id -> KalmanTrack
        self.next_id = 1
        self.dt = dt
        self.max_missed = max_missed_frames
        # 🔴 NUEVO: Memoria a largo plazo para cerdos dormidos
        self.retired_tracks = {}

    def predict_all(self):
        """Llama a la predicción en todos los tracks activos."""
        for t in self.tracks.values():
            t.predict()

    def remove_lost(self):
            """
            Elimina los tracks perdidos.
            🔴 CAMBIO: Antes de borrar, archiva su identidad visual.
            """
            to_del = []
            for tid, t in self.tracks.items():
                if t.consecutive_missed > self.max_missed:
                    to_del.append(tid)     
            for tid in to_del:
                # Archivar identidad antes de borrar
                track_to_retire = self.tracks[tid]
                if track_to_retire.prototype is not None:
                    self.retired_tracks[tid] = track_to_retire.prototype
                    print(f"💤 ID {tid} archivado en memoria.")

                del self.tracks[tid]
                print(f"Track ID {tid} eliminado de activos.")

    def create_track(self, bbox, emb):
            """
            Crea un nuevo track.
            🔴 CAMBIO: Intenta revivir un ID archivado si se parece mucho.
            """
            # 1. INTENTO DE RESURECCIÓN
            if emb is not None and len(self.retired_tracks) > 0:
                best_id = None
                best_sim = -1
                
                # Buscar en el archivo de retirados
                for ret_id, ret_emb in self.retired_tracks.items():
                    sim = float(cosine_similarity(emb, ret_emb.reshape(1, -1))[0])
                    
                    # Umbral de resurrección (0.80 es seguro)
                    if sim > 0.80:
                        if sim > best_sim:
                            best_sim = sim
                            best_id = ret_id
                
                # Si encontramos al cerdo antiguo
                if best_id is not None:
                    print(f"✨ ID {best_id} REVIVIÓ (Sim: {best_sim:.2f})")
                    
                    # Revivimos el track con el ID VIEJO
                    tr = KalmanTrack(best_id, bbox, emb, dt=self.dt)
                    
                    # Mezclamos la apariencia vieja con la nueva
                    old_emb = self.retired_tracks[best_id]
                    m = 0.5
                    tr.prototype = m * old_emb + (1 - m) * emb
                    tr.prototype /= (np.linalg.norm(tr.prototype) + 1e-6)
                    
                    self.tracks[best_id] = tr
                    
                    # Lo sacamos del archivo
                    del self.retired_tracks[best_id]
                    return tr

            # 2. CREACIÓN NORMAL
            tid = self.next_id
            self.next_id += 1
            tr = KalmanTrack(tid, bbox, emb, dt=self.dt)
            self.tracks[tid] = tr
            return tr

    def get_active_ids(self):
        """Devuelve los IDs activos."""
        return list(self.tracks.keys())
    def merge_and_heal(self, iou_threshold=0.1, reid_threshold=0.70, distance_threshold=200.0):
        fresh_ids = [tid for tid, t in self.tracks.items() if t.consecutive_missed == 0]
        stale_ids = [tid for tid, t in self.tracks.items() if t.consecutive_missed > 0]
        
        to_remove = set()

        for fresh_id in fresh_ids:
            if fresh_id in to_remove: continue

            fresh_track = self.tracks[fresh_id]
            fresh_box = fresh_track.get_predicted_bbox()
            fresh_center = center_of(fresh_box)
            fresh_emb = fresh_track.prototype

            best_match_id = None
            best_match_score = -1

            for stale_id in stale_ids:
                stale_track = self.tracks[stale_id]
                stale_box = stale_track.get_predicted_bbox()
                stale_center = center_of(stale_box)
                stale_emb = stale_track.prototype

                # Calcular métricas
                overlap = iou(fresh_box, stale_box)
                dist = np.sqrt((fresh_center[0]-stale_center[0])**2 + (fresh_center[1]-stale_center[1])**2)
                
                sim = 0.0
                if fresh_emb is not None and stale_emb is not None:
                    sim = float(cosine_similarity(fresh_emb, stale_emb.reshape(1, -1))[0])

                should_merge = False
                
                # CASO A: Solapamiento físico (Normal)
                if overlap > iou_threshold and sim > reid_threshold:
                    should_merge = True
                
                # CASO B: Proximidad (Cerdito se movió poco)
                elif dist < distance_threshold and sim > (reid_threshold + 0.05):
                    should_merge = True
                    
                # CASO C: TELEPORTACIÓN (Cerdito corrió lejos mientras estaba oculto)
                # Aquí ignoramos la distancia, pero exigimos una similitud MUY ALTA.
                # Esto evita confundirlo con otro cerdo lejano.
                elif sim > 0.88:  # <--- AJUSTA ESTE VALOR SI ES NECESARIO
                    should_merge = True
                    # print(f"🚀 TELEPORT: ID {stale_id} saltó {dist:.0f}px hacia ID {fresh_id}")

                if should_merge:
                    # Guardamos el que tenga mayor similitud visual
                    if sim > best_match_score:
                        best_match_score = sim
                        best_match_id = stale_id

            if best_match_id is not None:
                stale_track = self.tracks[best_match_id]
                
                # --- FUSIÓN ---
                # Traemos al ID viejo a la nueva posición
                stale_track.x = fresh_track.x.copy()
                stale_track.P = fresh_track.P.copy()
                stale_track.bbox = fresh_track.bbox
                stale_track.w = fresh_track.w
                stale_track.h = fresh_track.h
                stale_track.consecutive_missed = 0
                stale_track.last_seen = fresh_track.last_seen
                
                # Actualizar apariencia
                if fresh_track.prototype is not None:
                    m = 0.7
                    stale_track.prototype = m * stale_track.prototype + (1 - m) * fresh_track.prototype
                    stale_track.prototype /= (np.linalg.norm(stale_track.prototype) + 1e-6)

                to_remove.add(fresh_id)
                print(f"✅ FUSIÓN: ID {best_match_id} recuperado (Sim: {best_match_score:.2f})")

        for tid in to_remove:
            if tid in self.tracks:
                del self.tracks[tid]