import cv2
import numpy as np
import math
from collections import deque

# ==========================================
# 1. CONSTANTS I CONFIGURACIÓ
# ==========================================

REAL_COORDS = np.float32([
    [0,  0],
    [10, 0],
    [10, 20],
    [0,  20]
])

MIN_AREA = 1400
MAX_AREA = 17000
IOU_THRESHOLD = 0.30
MAX_LOST_FRAMES = 30
MAX_PLAYERS = 4

PLAYER_COLORS = [
    (0,   255, 100),   # J1 - verd
    (0,   100, 255),   # J2 - blau
    (255,  50,  50),   # J3 - vermell
    (180,   0, 255),   # J4 - lila
]

HEATMAP_COLORMAPS = [
    cv2.COLORMAP_SUMMER,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_SPRING,
]

ROI_EXCLUDE = []

def get_player_color(player_id):
    return PLAYER_COLORS[(player_id - 1) % len(PLAYER_COLORS)]

def get_player_colormap(player_id):
    return HEATMAP_COLORMAPS[(player_id - 1) % len(HEATMAP_COLORMAPS)]


# ==========================================
# 2. LÒGICA DE QUADRANTS I PROBABILITATS
# ==========================================

# Quadrant assignat a cada ID (1=sup-esq, 2=sup-dre, 3=inf-esq, 4=inf-dre)
# camp 10x20 -> y<10 és superior, y>=10 és inferior
#             -> x<5  és esquerra, x>=5 és dreta
PLAYER_QUADRANT = {
    1: ("top",    "left"),
    2: ("top",    "right"),
    3: ("bottom", "left"),
    4: ("bottom", "right"),
}

# Posició "home" per defecte de cada jugador (metres)
PLAYER_HOME = {
    1: (2.5,  5.0),   # sup-esq
    2: (7.5,  5.0),   # sup-dre
    3: (2.5, 15.0),   # inf-esq
    4: (7.5, 15.0),   # inf-dre
}


def get_real_quadrant(pos_m):
    """Retorna el quadrant (hemi vertical, hemi horitzontal) d'una posició en metres."""
    if pos_m is None:
        return None, None
    x, y = pos_m
    v = "top" if y < 10 else "bottom"
    h = "left" if x < 5 else "right"
    return v, h


def quadrant_probability(player_id, pos_m, last_pos_m,
                         max_dist=8.0, w_quadrant=10.0, w_side=2.0, w_dist=3.0):
    """
    Retorna una puntuació de probabilitat [0..inf) que la detecció a pos_m
    correspongui al jugador player_id.

    Factors:
      - Quadrant vertical obligatori: si la detecció no és al costat correcte
        (superior/inferior) la probabilitat és EXACTAMENT 0.
      - Costat horitzontal preferit: bonificació si coincideix, però no és
        eliminatoria (els jugadors es poden creuar dins del seu semipista).
      - Distància a l'última posició coneguda (o a la posició 'home').
    """
    if pos_m is None:
        return 0.0

    req_v, req_h = PLAYER_QUADRANT[player_id]
    det_v, det_h = get_real_quadrant(pos_m)

    # Restricció dura: vertical (sup/inf) no es pot creuar
    if det_v != req_v:
        return 0.0

    # Bonificació per costat horitzontal preferit
    side_score = w_side if det_h == req_h else 0.0

    # Distància a l'última posició (o posició home si no en tenim)
    ref = last_pos_m if last_pos_m is not None else PLAYER_HOME[player_id]
    dist = math.sqrt((pos_m[0] - ref[0])**2 + (pos_m[1] - ref[1])**2)
    dist_score = w_dist * max(0.0, 1.0 - dist / max_dist)

    return w_quadrant + side_score + dist_score


def assign_by_probability(lost_ids, unmatched_dets, homography, frame_cx_cy_fn):
    """
    Donats els IDs perduts i les deteccions sense parella, retorna la millor
    assignació {det_idx -> player_id} usant probabilitats per quadrant.

    homography: objecte Homography per projectar píxels -> metres
    frame_cx_cy_fn: funció(det) -> (cx_pixels, base_y_pixels) per cada detecció
    """
    if not lost_ids or not unmatched_dets:
        return {}

    # Construir matriu de scores [det x id]
    scores = {}
    for det_i, det in enumerate(unmatched_dets):
        cx, base_y = frame_cx_cy_fn(det)
        pos_m = homography.project_point(cx, base_y)

        for pid in lost_ids:
            last_pos = det.get("_last_pos_m")  # pot ser None
            scores[(det_i, pid)] = quadrant_probability(pid, pos_m, last_pos)

    # Assignació greedy: millor score primer
    assignment = {}
    used_dets = set()
    used_ids  = set()

    # Ordenar per score descendent
    sorted_pairs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    for (det_i, pid), score in sorted_pairs:
        if score <= 0:
            break  # la resta també seran 0 o pitjor
        if det_i in used_dets or pid in used_ids:
            continue
        assignment[det_i] = pid
        used_dets.add(det_i)
        used_ids.add(pid)

    return assignment


# ==========================================
# 3. CLASSE HOMOGRAFIA
# ==========================================
class Homography:
    def __init__(self):
        self.H = None
        self.pixel_points = []

    def compute_from_frame(self, frame):
        self.pixel_points = []
        clone = frame.copy()

        print("\n[HOMOGRAFIA] Clica els 4 cantons del TERRA de la pista.")
        print("IMPORTANT: Clica on la moqueta toca el vidre, NO a dalt del vidre!")
        print("Ordre: Sup-Esq -> Sup-Dreta -> Inf-Dreta -> Inf-Esq")

        cv2.namedWindow("Selecciona 4 punts del TERRA", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Selecciona 4 punts del TERRA", 1280, 720)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.pixel_points) < 4:
                self.pixel_points.append((x, y))
                print(f"Punt {len(self.pixel_points)}: ({x}, {y})")

        cv2.setMouseCallback("Selecciona 4 punts del TERRA", on_click)

        while True:
            display = clone.copy()
            for i, p in enumerate(self.pixel_points):
                cv2.circle(display, p, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (p[0]+10, p[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Selecciona 4 punts del TERRA", display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.pixel_points = []
            elif len(self.pixel_points) == 4:
                print("[HOMOGRAFIA] Punts seleccionats. Prem ENTER per confirmar.")
                if key in [13, 10]:
                    break

        cv2.destroyWindow("Selecciona 4 punts del TERRA")

        pts = np.float32(self.pixel_points)
        self.H, _ = cv2.findHomography(pts, REAL_COORDS)
        print("[HOMOGRAFIA] Matriu calculada correctament.\n")
        return self.H

    def project_point(self, px, py):
        if self.H is None:
            return None
        pt = np.float32([[px, py]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pt, self.H)
        return projected[0][0]


# ==========================================
# 4A. FILTRE DE ZONES ESTÀTIQUES (warmup)
# ==========================================
class StaticZoneFilter:
    def __init__(self, warmup_frames=90):
        self.warmup_frames = warmup_frames
        self.noise_mask    = None
        self.frame_count   = 0
        self.accumulator   = None

    def update(self, fg_mask):
        self.frame_count += 1
        if self.frame_count <= self.warmup_frames:
            if self.accumulator is None:
                self.accumulator = np.zeros_like(fg_mask, dtype=np.float32)
            self.accumulator += fg_mask.astype(np.float32)
            if self.frame_count == self.warmup_frames:
                avg = self.accumulator / self.warmup_frames
                _, self.noise_mask = cv2.threshold(
                    avg.astype(np.uint8), 200, 255, cv2.THRESH_BINARY
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                self.noise_mask = cv2.dilate(self.noise_mask, kernel, iterations=2)
                print(f"[FILTRE ESTÀTIC] Màscara de soroll calculada (frame {self.frame_count})")
            return fg_mask
        if self.noise_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(self.noise_mask))
        return fg_mask


# ==========================================
# 4B. FILTRE ROI
# ==========================================
class RoiFilter:
    def __init__(self, frame_shape, predefined_polys=None):
        h, w = frame_shape[:2]
        self.mask = np.ones((h, w), dtype=np.uint8) * 255
        if predefined_polys:
            for poly in predefined_polys:
                cv2.fillPoly(self.mask, [poly], 0)
            print(f"[ROI] {len(predefined_polys)} zones excloses carregades.")

    def apply(self, fg_mask):
        return cv2.bitwise_and(fg_mask, self.mask)


# ==========================================
# 4C. FILTRE DE CONSISTÈNCIA TEMPORAL
# ==========================================
class TemporalConsistencyFilter:
    def __init__(self, window=5, min_hits=3):
        self.window   = window
        self.min_hits = min_hits
        self.buffer   = deque(maxlen=window)

    def update(self, fg_mask):
        self.buffer.append(fg_mask.copy())
        if len(self.buffer) < self.window:
            return fg_mask
        stack = np.stack(list(self.buffer), axis=0).astype(np.uint16)
        count = (stack > 0).sum(axis=0).astype(np.uint8)
        _, consistent = cv2.threshold(count, self.min_hits - 1, 255, cv2.THRESH_BINARY)
        return consistent.astype(np.uint8)


# ==========================================
# 4D. PORTA DE CONTORNS
# ==========================================
def edge_gate(fg_mask, frame, low=30, high=90, dilate_px=15):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    return cv2.bitwise_and(fg_mask, edges_dilated)


# 5. TRACKER AMB ASSIGNACIÓ PROBABILÍSTICA
def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (aw*ah) + (bw*bh) - inter
    return inter / union if union > 0 else 0


class Tracker:
    """
    Tracker amb assignació per quadrant probabilística.

    - J1 i J2 només s'assignen a deteccions del semipista SUPERIOR (y_m < 10).
    - J3 i J4 només s'assignen a deteccions del semipista INFERIOR (y_m >= 10).
    - Dins de cada semipista, la preferència esquerra/dreta afecta el score
      però NO és eliminatòria (els jugadors es creuen).
    - Quan es perden jugadors, en lloc de LIFO s'usa assignació probabilística:
      es calcula un score per a cada parell (detecció, ID_perdut) i s'assigna
      el millor match per greedy.
    """

    def __init__(self, max_players=MAX_PLAYERS, homography=None):
        self.tracks       = []
        self.max_players  = max_players
        self.homo         = homography   # pot ser None fins que es configuri
        self.all_ids      = set(range(1, max_players + 1))
        # Darrera posició en metres coneguda per cada ID
        self._last_pos_m  = {pid: None for pid in self.all_ids}

    def set_homography(self, homography):
        self.homo = homography

    # ---- utilitats internes ----

    def _active_ids(self):
        return {t["id"] for t in self.tracks}

    def _lost_ids(self):
        return self.all_ids - self._active_ids()

    def _project(self, cx, base_y):
        if self.homo is None:
            return None
        return self.homo.project_point(cx, base_y)

    # ---- update principal ----

    def update(self, mask):
        # Morfologia per separar blobs verticals
        kernel_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
        mask_clean   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vert)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean   = cv2.erode(mask_clean, kernel_erode, iterations=1)

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / w if w > 0 else 0
                if aspect < 0.8:
                    continue
                detections.append({
                    "bbox":   (x, y, w, h),
                    "cx":     x + w // 2,
                    "base_y": y + h,
                })

        # Ordenar per àrea descendent i limitar a max_players
        detections = sorted(
            detections, key=lambda d: d["bbox"][2]*d["bbox"][3], reverse=True
        )[:self.max_players]

        matched_det_idx = set()

        # ---- Pas 1: aparellar pistes actives (IoU + distància) ----
        for t in self.tracks:
            best_det = None
            best_iou = IOU_THRESHOLD
            min_dist = 70

            for i, det in enumerate(detections):
                if i in matched_det_idx:
                    continue
                iou = compute_iou(t["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
                if best_det is None:
                    dist = math.sqrt(
                        (t["cx"]-det["cx"])**2 + (t["base_y"]-det["base_y"])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_det = i

            if best_det is not None:
                det = detections[best_det]
                t["bbox"]   = det["bbox"]
                t["cx"]     = det["cx"]
                t["base_y"] = det["base_y"]
                t["lost"]   = 0
                matched_det_idx.add(best_det)
                # Actualitzar posició en metres
                pos_m = self._project(t["cx"], t["base_y"])
                if pos_m is not None:
                    self._last_pos_m[t["id"]] = tuple(pos_m)
            else:
                t["lost"] += 1

        # ---- Pas 2: eliminar pistes massa perdudes ----
        surviving = []
        for t in self.tracks:
            if t["lost"] <= MAX_LOST_FRAMES:
                surviving.append(t)
            else:
                print(f"[TRACKER] ID {t['id']} perdut definitivament")
        self.tracks = surviving

        # ---- Pas 3: assignar deteccions sense parella a IDs perduts ----
        #   Aquí és on s'aplica la lògica probabilística per quadrant.
        unmatched = [
            (i, det) for i, det in enumerate(detections)
            if i not in matched_det_idx
        ]
        lost_ids = sorted(self._lost_ids())  # IDs sense pista activa

        if unmatched and lost_ids and self.homo is not None:
            # Preparar dets amb última posició coneguda (per al càlcul de distància)
            unmatched_dets_augmented = []
            for _, det in unmatched:
                aug = dict(det)
                # La darrera posició la guardem per quadrant_probability
                # (l'index det_i que rep assign_by_probability coincideix amb
                #  la posició dins unmatched_dets_augmented)
                aug["_last_pos_m"] = None  # no tenim info per a una nova pista
                unmatched_dets_augmented.append(aug)

            # Per als IDs perduts, injectem la darrera posició coneguda
            # a les dets (petita trampa: quadrant_probability la rep com a param)
            # Millor: fer-ho directament aquí amb la taula _last_pos_m
            def frame_cx_cy(det):
                return det["cx"], det["base_y"]

            # Puntuació per a cada parell (det_idx, player_id)
            scores = {}
            for det_i, aug_det in enumerate(unmatched_dets_augmented):
                cx, base_y = frame_cx_cy(aug_det)
                pos_m = self._project(cx, base_y)
                for pid in lost_ids:
                    last_pos = self._last_pos_m.get(pid)
                    scores[(det_i, pid)] = quadrant_probability(
                        pid, pos_m, last_pos
                    )

            # Assignació greedy: millor score primer
            used_dets = set()
            used_pids = set()
            active_ids = self._active_ids()

            for (det_i, pid), score in sorted(
                scores.items(), key=lambda kv: kv[1], reverse=True
            ):
                if score <= 0:
                    break
                if det_i in used_dets or pid in used_pids:
                    continue
                if len(active_ids) >= self.max_players:
                    break

                _, det = unmatched[det_i]
                pos_m = self._project(det["cx"], det["base_y"])

                self.tracks.append({
                    "id":     pid,
                    "bbox":   det["bbox"],
                    "cx":     det["cx"],
                    "base_y": det["base_y"],
                    "lost":   0,
                })
                if pos_m is not None:
                    self._last_pos_m[pid] = tuple(pos_m)

                active_ids.add(pid)
                used_dets.add(det_i)
                used_pids.add(pid)
                matched_det_idx.add(unmatched[det_i][0])

                print(
                    f"[TRACKER] ID {pid} reassignat per probabilitat "
                    f"(score={score:.2f}, pos_m={pos_m})"
                )

        return [t for t in self.tracks if t["lost"] == 0]

# 6. CLASSE HEATMAP PER JUGADOR
class PlayerHeatmap:
    def __init__(self, resolution=80):
        self.res     = resolution
        self.w_px    = 10 * resolution
        self.h_px    = 20 * resolution
        self.RADI_PX = int(resolution * 0.4)
        self.maps    = {}

    def _ensure_player(self, player_id):
        if player_id not in self.maps:
            self.maps[player_id] = np.zeros(
                (self.h_px, self.w_px), dtype=np.float32
            )

    def update(self, player_id, pos_m):
        if pos_m is None:
            return
        self._ensure_player(player_id)
        x_m, y_m = pos_m
        if -1 <= x_m <= 11 and -1 <= y_m <= 21:
            px = int(x_m * self.res)
            py = int(y_m * self.res)
            if 0 <= px < self.w_px and 0 <= py < self.h_px:
                cv2.circle(self.maps[player_id], (px, py), self.RADI_PX, 1.0, -1)

    def draw_court(self, img):
        r = self.res
        h, w = img.shape[:2]
        white = (255, 255, 255)
        cv2.rectangle(img, (0, 0), (w-1, h-1), white, 3)
        cv2.line(img, (0, h // 2), (w, h // 2), white, 4)
        dist_servei_m = 6.95
        y_serv_sup = int((10 - dist_servei_m) * r)
        y_serv_inf = int((10 + dist_servei_m) * r)
        cv2.line(img, (0, y_serv_sup), (w, y_serv_sup), white, 2)
        cv2.line(img, (0, y_serv_inf), (w, y_serv_inf), white, 2)
        cv2.line(img, (w // 2, y_serv_sup), (w // 2, y_serv_inf), white, 2)

    def _process_map(self, raw_map):
        blurred = cv2.GaussianBlur(raw_map, (41, 41), 0)
        p95 = np.percentile(blurred, 95)
        if p95 == 0:
            p95 = blurred.max()
        if p95 == 0:
            return None
        clipped = np.clip(blurred, 0, p95)
        return (clipped / p95 * 255).astype(np.uint8)

    def save_individual(self, base_path="heatmap"):
        for pid, raw_map in self.maps.items():
            norm = self._process_map(raw_map)
            if norm is None:
                continue
            colored = cv2.applyColorMap(norm, get_player_colormap(pid))
            self.draw_court(colored)
            path = f"{base_path}_jugador_{pid}.png"
            cv2.imwrite(path, colored)
            print(f"[HEATMAP] Jugador {pid} guardat a: {path}")

    def save_combined(self, path="heatmap_combinat.png"):
        if not self.maps:
            print("[HEATMAP] Cap dada. No es guarda.")
            return
        combined   = np.zeros((self.h_px, self.w_px, 3), dtype=np.float32)
        weight_sum = np.zeros((self.h_px, self.w_px), dtype=np.float32)
        for pid, raw_map in self.maps.items():
            norm = self._process_map(raw_map)
            if norm is None:
                continue
            bgr       = get_player_color(pid)
            intensity = norm.astype(np.float32) / 255.0
            for c, channel_val in enumerate(bgr):
                combined[:, :, c] += intensity * channel_val
            weight_sum += intensity
        mask = weight_sum > 0
        for c in range(3):
            combined[:, :, c][mask] = np.clip(
                combined[:, :, c][mask] / weight_sum[mask] * 255, 0, 255
            )
        result   = combined.astype(np.uint8)
        legend_h = 50
        legend   = np.zeros((legend_h, self.w_px, 3), dtype=np.uint8)
        for pid in sorted(self.maps.keys()):
            x_leg = int((pid - 1) * 100)
            if x_leg + 90 > self.w_px:
                break
            color = get_player_color(pid)
            cv2.rectangle(legend, (x_leg + 5, 10), (x_leg + 25, 35), color, -1)
            cv2.putText(legend, f"J{pid}", (x_leg + 30, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        final = np.vstack([result, legend])
        self.draw_court(final)
        cv2.imwrite(path, final)
        print(f"[HEATMAP] Combinat guardat a: {path}")

# EXECUCIÓ PRINCIPAL (MAIN)

if __name__ == "__main__":
    video_path = "Data-Set/padel-data-labels/2022_BCN_FinalM_Retallat_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error obrint el vídeo: {video_path}")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("El vídeo està buit.")
        exit()

    homo      = Homography()
    heatmap   = PlayerHeatmap(resolution=80)
    tracker   = Tracker(max_players=MAX_PLAYERS)

    static_filter   = StaticZoneFilter(warmup_frames=90)
    roi_filter      = RoiFilter(frame.shape, predefined_polys=ROI_EXCLUDE)
    temporal_filter = TemporalConsistencyFilter(window=5, min_hits=3)

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=1700,
        varThreshold=100,
        detectShadows=True
    )


    H = homo.compute_from_frame(frame)
    if H is None:
        print("S'ha cancel·lat la selecció de l'homografia.")
        exit()

    # Ara que tenim l'homografia, la passem al tracker
    tracker.set_homography(homo)

    print("\n[INFO] Processant el vídeo... Prem 'q' per aturar-ho.")

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = roi_filter.apply(fg_mask)
        fg_mask = static_filter.update(fg_mask)
        fg_mask = temporal_filter.update(fg_mask)
        fg_mask = edge_gate(fg_mask, frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

        active_tracks = tracker.update(fg_mask)

        for t in active_tracks:
            pid   = t["id"]
            color = get_player_color(pid)
            x, y, w, h = t["bbox"]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label = f"J{pid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y - lh - 10), (x + lw + 8, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            cv2.circle(frame, (t["cx"], t["base_y"]), 5, color, -1)

            pos_m = homo.project_point(t["cx"], t["base_y"])
            heatmap.update(pid, pos_m)

        cv2.imshow("Tracking Padel", frame)
        cv2.imshow("Mascara Jugadors", fg_mask)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    heatmap.save_individual("heatmap_padel")
    heatmap.save_combined("heatmap_padel_combinat.png")