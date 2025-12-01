# tracking/matcher.py

import numpy as np
import config
from reid.reid_helper import cosine_similarity
from utils.iou_utils import center_of, iou


def perform_matching(manager, detections, img_diag):
    """
    Realiza el matching greedy entre tracks activos y nuevas detecciones
    usando un costo combinado (apariencia + movimiento).
    """
    unmatched_dets = set(range(len(detections)))
    unmatched_tracks = set(manager.get_active_ids())
    matched_pairs = []
    track_list = list(manager.tracks.keys())
    
    if not track_list or not detections:
        return matched_pairs, unmatched_tracks, unmatched_dets

    # 1. Calcular Matriz de Costos
    costs = []
    for tid in track_list:
        tr = manager.tracks[tid]
        row = []
        for det in detections:
            motion_cost = tr.motion_distance(det["bbox"], img_diag)
            
            sim = 0.0
            if det["emb"] is not None and tr.prototype is not None:
                sim = float(cosine_similarity(det["emb"], tr.prototype.reshape(1, -1))[0])
            
            app_cost = 1.0 - sim
            combined_cost = config.ALPHA * app_cost + config.BETA * motion_cost
            row.append(combined_cost)
        costs.append(row)
    
    costs = np.array(costs)
    cost_copy = costs.copy()

    # 2. Matching Greedy con Comprobaciones de Puerta
    while True:
        if np.all(np.isinf(cost_copy)):
            break
        ti, di = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
        minval = cost_copy[ti, di]
        if np.isinf(minval):
            break
        
        tid = track_list[ti]
        tr = manager.tracks[tid]
        det = detections[di]

        # Re-calcular métricas de la mejor pareja
        motion_norm = tr.motion_distance(det["bbox"], img_diag)
        sim = 0.0
        if det["emb"] is not None and tr.prototype is not None:
            sim = float(cosine_similarity(det["emb"], tr.prototype.reshape(1, -1))[0])

        cx_pred, cy_pred = tr.x[0], tr.x[1]
        cx_det, cy_det = center_of(det["bbox"])
        frame_dist = np.sqrt((cx_pred - cx_det) ** 2 + (cy_pred - cy_det) ** 2)

        # Comprobación de Teleport (demasiado lejos Y apariencia no confiable)
        reject_teleport = (
            frame_dist > config.MAX_SPEED_PIXELS_PER_FRAME
            and sim < (config.REID_THRESHOLD + 0.05)
        )
        
        # Criterio de Aceptación: movimiento pequeño O apariencia buena
        accept = False
        if not reject_teleport:
            accept = (motion_norm <= config.MOTION_GATE) or (sim >= (config.REID_THRESHOLD - 0.08))

        # Comprobación de oclusión (si muchos tracks predichos se solapan con esta det, requiere más confianza)
        if accept:
            overlapping_tracks = []
            for other_tid, other_tr in manager.tracks.items():
                if other_tid != tid and other_tr in unmatched_tracks:  # Solo tracks que aún no están matcheados
                    pred_box = other_tr.get_predicted_bbox()
                    if iou(pred_box, det["bbox"]) > config.OCCLUSION_IOU:
                        overlapping_tracks.append(other_tid)
            
            if len(overlapping_tracks) > 0 and sim < (config.REID_THRESHOLD + 0.05):
                # Ambiguo: si no es muy confiable, rechazar match para esta detección.
                accept = False

        if accept:
            matched_pairs.append((tid, di, minval))
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        # Invalidar fila y columna para el siguiente paso greedy
        cost_copy[ti, :] = np.inf
        cost_copy[:, di] = np.inf

    return matched_pairs, unmatched_tracks, unmatched_dets