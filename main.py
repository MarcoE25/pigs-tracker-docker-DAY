# main.py

import time
import os
import cv2
import numpy as np
import csv

# Importar módulos
import config
from detection.detection_yolo import PigdetectorYOLO
from tracking.kalman_track import KalmanTrack
from tracking.tracker_manager import TrackerManager
from tracking.matcher import perform_matching
from utils.filter_utils import filter_and_process_detections
from utils.draw_utils import draw_tracks
from reid.reid_helper import ReExtractor, save_gallery, load_gallery, cosine_similarity

def setup_resources():
    """Inicializa modelos, managers y recursos de video."""
    print("--- Configuración Inicial ---")
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {config.MODEL_PATH}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    try:
        detector = PigdetectorYOLO()
    except NameError:
        from detection.detection_yolo import PigdetectorYOLO
        detector = PigdetectorYOLO()

    reid = ReExtractor()
    print("ReID device:", reid.device)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"❌ No se pudo abrir video: {config.VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS_GUESS
    dt = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_diag = np.sqrt(width**2 + height**2)
    MAX_MISSED_FRAMES = int(config.MAX_MISSED_SECONDS * fps)

    manager = TrackerManager(dt=dt, max_missed_frames=MAX_MISSED_FRAMES)

    out = None
    if config.SAVE_VIDEO:
        out = cv2.VideoWriter(
            os.path.join(config.OUTPUT_DIR, "tracked_reid_kalman.avi"),
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (width, height)
        )
        print("Guardando video de salida...")

    return detector, reid, cap, out, manager, width, height, img_diag, fps


def center_of(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def process_frame(frame, frame_count, current_time, elapsed,
                  detector, reid, manager, width, height, img_diag,
                  gallery, gallery_built, warmup_embeddings,
                  last_seen, lost_alert_sent, track_counts):

    manager.predict_all()
    results = detector.detect_and_track(frame)

    detections = []
    if results and len(results) > 0 and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        detections = filter_and_process_detections(boxes, ids, confs, frame, reid)

    if (not gallery_built) and elapsed <= config.WARMUP_SECONDS:
        for det in detections:
            if det["emb"] is None:
                continue
            tid = det["tracker_id"]
            warmup_embeddings.setdefault(tid, []).append(det["emb"])

    matched_pairs, unmatched_tracks, unmatched_dets = perform_matching(
        manager, detections, img_diag
    )

    for tid, di, _ in matched_pairs:
        det = detections[di]
        tr = manager.tracks[tid]

        cx_pred, cy_pred = tr.x[0], tr.x[1]
        cx_det, cy_det = center_of(det["bbox"])
        frame_dist = np.sqrt((cx_pred - cx_det) ** 2 + (cy_pred - cy_det) ** 2)

        sim = 0.0
        if det["emb"] is not None and tr.prototype is not None:
            sim = float(cosine_similarity(det["emb"], tr.prototype.reshape(1, -1))[0])

        if frame_dist > config.MAX_SPEED_PIXELS_PER_FRAME and sim < (config.REID_THRESHOLD + 0.05):
            tr.mark_missed()
        else:
            tr.update(det["bbox"], det["emb"])
            last_seen[tid] = current_time
            lost_alert_sent[tid] = False
            track_counts[tid] = track_counts.get(tid, 0) + 1

    for di in list(unmatched_dets):
        det = detections[di]
        assigned = False

        if gallery_built and det["emb"] is not None:
            gallery_ids = list(gallery.keys())
            gallery_embs = np.vstack([gallery[k] for k in gallery_ids])
            sims = cosine_similarity(det["emb"], gallery_embs)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_gid = gallery_ids[best_idx]

            if best_sim >= config.REID_THRESHOLD:
                tr = manager.create_track(det["bbox"], gallery[best_gid].copy())
                tr.prototype = gallery[best_gid].copy()
                assigned = True

        if not assigned:
            x1, y1, x2, y2 = det["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area > 100:
                tr = manager.create_track(det["bbox"], det["emb"])
                last_seen[tr.id] = current_time
                lost_alert_sent[tr.id] = False
                track_counts[tr.id] = track_counts.get(tr.id, 0) + 1

    for tid in list(unmatched_tracks):
        if tid in manager.tracks:
            manager.tracks[tid].mark_missed()

    # --- NUEVA LÓGICA: LIMPIEZA DE FANTASMAS ---
    # Usamos merge_and_heal en lugar de clean_duplicates
    if hasattr(manager, "merge_and_heal"):
        # Umbrales sugeridos: iou=0.3 (toque leve) y reid=0.80 (bastante parecido)
        manager.merge_and_heal(iou_threshold=0.3, reid_threshold=0.80)
    # -------------------------------------------

    manager.remove_lost()

    return gallery, gallery_built, warmup_embeddings, last_seen, lost_alert_sent, track_counts


def run_pig_tracking():
    try:
        detector, reid, cap, out, manager, width, height, img_diag, fps = setup_resources()
    except (FileNotFoundError, IOError, NameError) as e:
        print(f"Error en setup: {e}")
        return

    start_time = time.time()
    frame_count = 0
    gallery = load_gallery(config.GALLERY_PATH)
    gallery_built = len(gallery) > 0
    warmup_embeddings = {}
    last_seen = {}
    lost_alert_sent = {}
    track_counts = {}

    print("--- Bucle de Procesamiento ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - start_time

        gallery, gallery_built, warmup_embeddings, last_seen, lost_alert_sent, track_counts = process_frame(
            frame, frame_count, current_time, elapsed,
            detector, reid, manager, width, height, img_diag,
            gallery, gallery_built, warmup_embeddings,
            last_seen, lost_alert_sent, track_counts
        )

        if (not gallery_built) and (elapsed > config.WARMUP_SECONDS):
            gallery = {
                tid: np.mean(np.vstack(embs), axis=0) /
                     (np.linalg.norm(np.mean(np.vstack(embs), axis=0)) + 1e-6)
                for tid, embs in warmup_embeddings.items()
                if len(embs) > 0
            }
            if len(gallery) > 0:
                gallery_built = True
                save_gallery(gallery, config.GALLERY_PATH)
                print(f"✅ Galería construida con {len(gallery)} entradas.")

        draw_tracks(frame, manager.tracks, width, height)

        for tid, last in list(last_seen.items()):
            time_missing = current_time - last
            if time_missing > config.ID_LOST_THRESHOLD and not lost_alert_sent.get(tid, False):
                print(f"⚠️ ALERTA: Cerdo ID {tid} perdido desde hace {time_missing:.1f} s")
                lost_alert_sent[tid] = True

        if config.SAVE_VIDEO:
            out.write(frame)
        
        # --- SECCIÓN COMENTADA PARA EVITAR ERROR DE GUI EN ENTORNO SIN CABEZA ---
        # Si arreglas tu OpenCV, puedes descomentar estas líneas:
        # cv2.imshow("ReID + Kalman Tracking (improved)", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        # -----------------------------------------------------------------------

    duration = time.time() - start_time
    avg_fps = frame_count / (duration + 1e-6)
    print("--- Resumen Final ---")
    print(f"Duración (s): {duration:.2f}, Frames: {frame_count}, FPS medio: {avg_fps:.2f}")

    csv_path = os.path.join(config.OUTPUT_DIR, "tracking_summary_kalman.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Frames"])
        for tid, cnt in sorted(track_counts.items()):
            writer.writerow([tid, cnt])
    print("CSV guardado en:", csv_path)

    if gallery_built:
        save_gallery(gallery, config.GALLERY_PATH)

    cap.release()
    if config.SAVE_VIDEO and out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pig_tracking()