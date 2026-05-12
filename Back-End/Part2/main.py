import cv2
import numpy as np
import math
import sys, os
from ultralytics import YOLO

# Afegim la Part_1 al path per poder importar les seves classes
import importlib.util, pathlib

_spec = importlib.util.spec_from_file_location(
    "main_part1",
    pathlib.Path(__file__).parent.parent / "Part_1" / "main.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

Homography           = _m.Homography
Tracker              = _m.Tracker
PlayerHeatmap        = _m.PlayerHeatmap
get_player_color     = _m.get_player_color
MAX_PLAYERS          = _m.MAX_PLAYERS
ROI_EXCLUDE          = _m.ROI_EXCLUDE
compute_iou          = _m.compute_iou
IOU_THRESHOLD        = _m.IOU_THRESHOLD
MAX_LOST_FRAMES      = _m.MAX_LOST_FRAMES
quadrant_probability = _m.quadrant_probability
PLAYER_HOME          = _m.PLAYER_HOME

# ==========================================
# CONFIGURACIÓ
# ==========================================

# Ruta al vídeo, relativa a la carpeta Part_2
VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 
    'Data-Set', 'padel-data-labels', 
    '2022_BCN_FinalM_Retallat_1.mp4'
)

# Model YOLO a utilitzar:
#   yolov8n.pt nano,  més ràpid  però menys precís  (recomanat per CPU)
#   yolov8s.pt small, equilibrat
#   yolov8m.pt medium, més precís però més lent     (recomanat amb GPU)
# El fitxer es descarrega automàticament la primera vegada
YOLO_MODEL = "yolov8n.pt"

# Confiança mínima de YOLO per acceptar una detecció (0.0 - 1.0)
# Valors baixos detecta més però amb més falsos positius
# Valors alts menys deteccions però més fiables
CONF_THRESHOLD = 0.4

# Llindar IoU per al NMS (Non-Maximum Suppression) intern de YOLO
# Controla quan dues caixes solapades es fusionen en una sola
# Valors baixos elimina més duplicats
# Valors alts permet més solapament entre deteccions
IOU_NMS = 0.5


# ==========================================
# TRACKER ESTÈS PER A YOLO
# ==========================================

class TrackerYOLO(Tracker):
    def update_from_dets(self, detections):
        detections = sorted(
            detections, key=lambda d: d["bbox"][2]*d["bbox"][3], reverse=True
        )[:self.max_players]

        matched_det_idx = set()

        for t in self.tracks:
            best_det, best_iou, min_dist = None, IOU_THRESHOLD, 70
            for i, det in enumerate(detections):
                if i in matched_det_idx:
                    continue
                iou = compute_iou(t["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou, best_det = iou, i
                if best_det is None:
                    dist = math.sqrt(
                        (t["cx"]-det["cx"])**2 + (t["base_y"]-det["base_y"])**2)
                    if dist < min_dist:
                        min_dist, best_det = dist, i
            if best_det is not None:
                det = detections[best_det]
                t.update({"bbox": det["bbox"], "cx": det["cx"],
                          "base_y": det["base_y"], "lost": 0})
                matched_det_idx.add(best_det)
                pos_m = self._project(t["cx"], t["base_y"])
                if pos_m is not None:
                    self._last_pos_m[t["id"]] = tuple(pos_m)
                self._buffer.cancel(t["id"])
            else:
                t["lost"] += 1

        self.tracks = [t for t in self.tracks if t["lost"] <= MAX_LOST_FRAMES]

        unmatched = [d for i, d in enumerate(detections)
                     if i not in matched_det_idx]
        lost_ids = sorted(self._lost_ids())

        if unmatched and lost_ids and self.homo is not None:
            def score_fn(pid, pos_m):
                return quadrant_probability(pid, pos_m, self._last_pos_m.get(pid))
            confirmed = self._buffer.process(
                lost_ids=lost_ids,
                unmatched_dets=unmatched,
                score_fn=score_fn,
                project_fn=self._project,
            )
            active_ids = self._active_ids()
            for pid, det in confirmed:
                if len(active_ids) >= self.max_players:
                    break
                pos_m = self._project(det["cx"], det["base_y"])
                self.tracks.append({
                    "id": pid, "bbox": det["bbox"],
                    "cx": det["cx"], "base_y": det["base_y"], "lost": 0,
                })
                if pos_m is not None:
                    self._last_pos_m[pid] = tuple(pos_m)
                active_ids.add(pid)

        return [t for t in self.tracks if t["lost"] == 0]


# ==========================================
# FILTRE ROI
# ==========================================

def build_roi_mask(shape):
    """Crea una màscara negra a les zones que volem ignorar
    (marcador, publicitat, cristalls) definides a la Part_1."""
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for poly in ROI_EXCLUDE:
        cv2.fillPoly(mask, [poly], 0)
    return mask

def detection_in_roi(bbox, roi_mask):
    """Comprova si el peu del jugador (base de la bbox) és dins la zona vàlida."""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h
    if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
        return roi_mask[cy, cx] > 0
    return False


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error obrint el vídeo: {VIDEO_PATH}"); exit()

    ret, frame = cap.read()
    if not ret:
        print("Vídeo buit."); exit()

    roi_mask = build_roi_mask(frame.shape)
    homo     = Homography()
    tracker  = TrackerYOLO(max_players=MAX_PLAYERS)
    heatmap  = PlayerHeatmap(resolution=80)

    H = homo.compute_from_frame(frame)
    if H is None: exit()
    tracker.set_homography(homo)

    print("\n[INFO] Processant amb YOLOv8... Prem 'q' per aturar.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inferència YOLO: detecta persones (classe 0 de COCO)
        results = model(
            frame, classes=[0],
            conf=CONF_THRESHOLD, iou=IOU_NMS, verbose=False,
        )[0]

        # Filtrar deteccions: fora de ROI i formes massa horitzontals
        yolo_dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = x1, y1, x2-x1, y2-y1
            if not detection_in_roi((x, y, w, h), roi_mask):
                continue
            if w > 0 and (h / w) < 0.8:
                continue
            yolo_dets.append({
                "bbox":   (x, y, w, h),
                "cx":     x + w // 2,
                "base_y": y + h,
            })

        active_tracks = tracker.update_from_dets(yolo_dets)
        ghost_tracks  = tracker.get_ghost_tracks()

        for t in active_tracks:
            pid   = t["id"]
            color = get_player_color(pid)
            x, y, w, h = t["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"J{pid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-lh-10), (x+lw+8, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label, (x+4, y-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.circle(frame, (t["cx"], t["base_y"]), 5, color, -1)
            pos_m = homo.project_point(t["cx"], t["base_y"])
            heatmap.update(pid, pos_m)

        cv2.putText(frame, f"YOLOv8 | persones: {len(results.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        
        # Generar màscara visual amb les deteccions YOLO
        debug_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for det in yolo_dets:
            x, y, w, h = det["bbox"]
            cv2.rectangle(debug_mask, (x, y), (x+w, y+h), 255, -1)

        cv2.imshow("Tracking Padel - YOLOv8", frame)
        cv2.imshow("Mascara YOLO", debug_mask)  # <-- afegeix això
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        cv2.imshow("Tracking Padel - YOLOv8", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    heatmap.save_individual("E:\VC-PG\Padel_Vision\Out\YOLO\heatmap_yolo")
    heatmap.save_combined("E:\VC-PG\Padel_Vision\Out\YOLO\heatmap_yolo_combinat.png")