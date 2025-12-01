# utils/draw_utils.py

import cv2

def draw_tracks(frame, tracks, width, height):
    """Dibuja las bounding boxes y IDs de los tracks activos."""
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr.get_predicted_bbox()
        
        # Asegurar que las coordenadas estén dentro del frame
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
        
        label = f"ID {tid} (missed: {tr.consecutive_missed})"
        color = (0, 200, 0)  # Verde
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            label,
            (int(x1), max(15, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
