# utils/filter_utils.py

import numpy as np
from utils.iou_utils import iou
import config
from reid.reid_helper import ReExtractor  # Se necesita importar desde el módulo ReID


def filter_and_process_detections(boxes, ids, confs, frame, reid_extractor):
    """
    1. Aplica filtros de área y aspecto.
    2. Fusiona cajas superpuestas (NMS suavizado).
    3. Calcula embeddings de ReID para las cajas resultantes.
    Devuelve una lista de diccionarios (detecciones limpias).
    """
    # 1) FILTRO POR ÁREA Y ASPECTO
    candidates = []
    for box, trk_id, conf in zip(boxes, ids, confs):
        x1, y1, x2, y2 = map(int, box)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect = w / float(h)

        if area < config.MIN_AREA:
            continue
        if aspect < config.MIN_ASPECT_RATIO or aspect > config.MAX_ASPECT_RATIO:
            continue

        candidates.append((x1, y1, x2, y2, int(trk_id), float(conf)))

    # 2) MERGE BOXES
    merged = []
    used = set()
    for i in range(len(candidates)):
        if i in used:
            continue
        a = candidates[i]
        group = [a]
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            b = candidates[j]
            if iou(a[:4], b[:4]) > config.IOU_MERGE:
                group.append(b)
                used.add(j)
        used.add(i)

        if len(group) == 1:
            merged.append(group[0])
        else:
            # Promedio de coordenadas
            mx1 = int(np.mean([g[0] for g in group]))
            my1 = int(np.mean([g[1] for g in group]))
            mx2 = int(np.mean([g[2] for g in group]))
            my2 = int(np.mean([g[3] for g in group]))
            rep = sorted(group, key=lambda g: g[5], reverse=True)[0]
            merged.append((mx1, my1, mx2, my2, rep[4], rep[5]))

    # 3) CALCULAR EMBEDDINGS
    detections = []
    for (x1, y1, x2, y2, trk_id, conf) in merged:
        pad = 4
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[ya:yb, xa:xb]
        try:
            emb = reid_extractor.get_embedding(crop)
        except Exception:
            emb = None

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "tracker_id": int(trk_id),
            "conf": float(conf),
            "emb": emb
        })
    return detections