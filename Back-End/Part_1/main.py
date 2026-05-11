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

MIN_AREA        = 1000
MAX_AREA        = 16000
IOU_THRESHOLD   = 0.30
MAX_LOST_FRAMES = 80
MAX_PLAYERS     = 4

PLAYER_COLORS = [
    (0,   255, 100),
    (0,   100, 255),
    (255,  50,  50),
    (180,   0, 255),
]

HEATMAP_COLORMAPS = [
    cv2.COLORMAP_SUMMER,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_SPRING,
]

ROI_EXCLUDE = []

# Paràmetres del buffer de confiança
MIN_SCORE_TO_TRACK   = 8.0   # score mínim per frame per crear candidatura
CONFIRM_THRESHOLD    = 45.0  # score acumulat per confirmar assignació
MAX_CANDIDATE_FRAMES = 20    # frames màxims que una candidatura pot esperar
CANDIDATE_MAX_DIST   = 55    # distància màx (px) per seguir la mateixa candidatura

def get_player_color(player_id):
    return PLAYER_COLORS[(player_id - 1) % len(PLAYER_COLORS)]

def get_player_colormap(player_id):
    return HEATMAP_COLORMAPS[(player_id - 1) % len(HEATMAP_COLORMAPS)]


# ==========================================
# 2. QUADRANTS I PROBABILITATS
# ==========================================

PLAYER_QUADRANT = {
    1: ("top",    "left"),
    2: ("top",    "right"),
    3: ("bottom", "left"),
    4: ("bottom", "right"),
}

PLAYER_HOME = {
    1: (2.5,  5.0),
    2: (7.5,  5.0),
    3: (2.5, 15.0),
    4: (7.5, 15.0),
}

def get_real_quadrant(pos_m):
    if pos_m is None:
        return None, None
    x, y = pos_m
    return ("top" if y < 10 else "bottom"), ("left" if x < 5 else "right")

def quadrant_probability(player_id, pos_m, last_pos_m,
                         max_dist=8.0, w_quadrant=10.0, w_side=2.0, w_dist=3.0):
    if pos_m is None:
        return 0.0
    req_v, req_h = PLAYER_QUADRANT[player_id]
    det_v, det_h = get_real_quadrant(pos_m)
    if det_v != req_v:
        return 0.0
    side_score = w_side if det_h == req_h else 0.0
    ref  = last_pos_m if last_pos_m is not None else PLAYER_HOME[player_id]
    dist = math.sqrt((pos_m[0] - ref[0])**2 + (pos_m[1] - ref[1])**2)
    return w_quadrant + side_score + w_dist * max(0.0, 1.0 - dist / max_dist)


# ==========================================
# 3. HOMOGRAFIA
# ==========================================

class Homography:
    def __init__(self):
        self.H     = None
        self.H_inv = None
        self.pixel_points = []

    def compute_from_frame(self, frame):
        self.pixel_points = []
        clone = frame.copy()
        print("\n[HOMOGRAFIA] Clica els 4 cantons del TERRA de la pista.")
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
        self.H, _  = cv2.findHomography(pts, REAL_COORDS)
        self.H_inv = np.linalg.inv(self.H)
        print("[HOMOGRAFIA] Matriu calculada.\n")
        return self.H

    def project_point(self, px, py):
        if self.H is None:
            return None
        pt = np.float32([[px, py]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pt, self.H)[0][0]

    def project_point_inv(self, xm, ym):
        if self.H_inv is None:
            return None
        pt = np.float32([[xm, ym]]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pt, self.H_inv)[0][0].astype(int)


# ==========================================
# 4. FILTRES
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
                    avg.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                self.noise_mask = cv2.dilate(self.noise_mask, kernel, iterations=2)
                print(f"[FILTRE ESTÀTIC] Màscara calculada (frame {self.frame_count})")
            return fg_mask
        if self.noise_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(self.noise_mask))
        return fg_mask


class RoiFilter:
    def __init__(self, frame_shape, predefined_polys=None):
        h, w = frame_shape[:2]
        self.mask = np.ones((h, w), dtype=np.uint8) * 255
        if predefined_polys:
            for poly in predefined_polys:
                cv2.fillPoly(self.mask, [poly], 0)

    def apply(self, fg_mask):
        return cv2.bitwise_and(fg_mask, self.mask)


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


def edge_gate(fg_mask, frame, low=30, high=90, dilate_px=15):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
    return cv2.bitwise_and(fg_mask, cv2.dilate(edges, kernel, iterations=2))


# ==========================================
# 5. MOG2 DUAL + DETECTOR DE PAUSA + MEDIANA
# ==========================================

class DualRateMOG2:
    def __init__(self, history_play=400, var_threshold=60,
                 lr_play=0.003, lr_pause=0.0, motion_thresh=600):
        self.lr_play       = lr_play
        self.lr_pause      = lr_pause
        self.motion_thresh = motion_thresh
        self.is_playing    = False
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=history_play, varThreshold=var_threshold, detectShadows=True)

    def apply(self, frame, motion_pixels):
        self.is_playing = motion_pixels > self.motion_thresh
        lr = self.lr_play if self.is_playing else self.lr_pause
        fg = self.mog.apply(frame, learningRate=lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        return fg


class PauseDetector:
    def __init__(self, window=25, play_thresh=600, pause_thresh=250, cooldown=12):
        self.play_thresh  = play_thresh
        self.pause_thresh = pause_thresh
        self.cooldown     = cooldown
        self._buffer      = deque(maxlen=window)
        self._state       = "play"
        self._since       = 0
        self.on_play_start = None

    def update(self, fg_mask):
        self._buffer.append(int(np.count_nonzero(fg_mask)))
        self._since += 1
        avg = float(np.mean(self._buffer)) if self._buffer else 0
        if self._state == "play":
            if avg < self.pause_thresh and self._since >= self.cooldown:
                self._state = "pause"
                self._since = 0
        elif self._state == "pause":
            if avg > self.play_thresh and self._since >= self.cooldown:
                self._state = "play"
                self._since = 0
                if self.on_play_start:
                    self.on_play_start()

    @property
    def motion_pixels(self):
        return float(np.mean(self._buffer)) if self._buffer else 0


class MedianBackground:
    def __init__(self, n_frames=150, threshold=18, skip=2):
        self.n_frames  = n_frames
        self.threshold = threshold
        self.skip      = skip
        self._frames   = []
        self._count    = 0
        self.background = None
        self.ready      = False

    def feed(self, frame):
        if self.ready:
            return True
        self._count += 1
        if self._count % self.skip != 0:
            return False
        self._frames.append(frame.astype(np.float32))
        if len(self._frames) >= self.n_frames:
            print(f"[MEDIAN BG] Calculant sobre {len(self._frames)} frames...")
            self.background = np.median(
                np.stack(self._frames, axis=0), axis=0).astype(np.uint8)
            self.ready = True
            self._frames.clear()
            print("[MEDIAN BG] Llest.")
        return self.ready

    def apply(self, frame):
        if not self.ready:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        diff = cv2.absdiff(frame, self.background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        return mask

    def save(self, path):
        if self.ready:
            cv2.imwrite(path, self.background)
            print(f"[MEDIAN BG] Guardat: {path}")

    def load(self, path):
        bg = cv2.imread(path)
        if bg is not None:
            self.background = bg
            self.ready = True
            print(f"[MEDIAN BG] Carregat: {path}")


# ==========================================
# 6. BUFFER DE CONFIANÇA PER A ASSIGNACIÓ
# ==========================================

class CandidateTrack:
    def __init__(self, player_id, det, score, pos_m):
        self.player_id         = player_id
        self.score_acc         = score
        self.frames_seen       = 1
        self.frames_since_seen = 0
        self.cx     = det["cx"]
        self.base_y = det["base_y"]
        self.pos_m  = pos_m
        self.det    = det

    def update(self, det, score, pos_m):
        self.score_acc         += score
        self.frames_seen       += 1
        self.frames_since_seen  = 0
        self.cx     = det["cx"]
        self.base_y = det["base_y"]
        self.pos_m  = pos_m
        self.det    = det

    def mark_missing(self):
        self.frames_since_seen += 1
        if self.frames_since_seen > 3:
            self.score_acc *= 0.7

    @property
    def confirmed(self):
        return self.score_acc >= CONFIRM_THRESHOLD

    @property
    def expired(self):
        return (self.frames_seen + self.frames_since_seen > MAX_CANDIDATE_FRAMES
                or (self.score_acc < MIN_SCORE_TO_TRACK * 2 and self.frames_seen > 5))

    @property
    def pct(self):
        return min(100, int(self.score_acc / CONFIRM_THRESHOLD * 100))


class AssignmentBuffer:
    def __init__(self):
        self._candidates: dict[int, CandidateTrack] = {}

    def process(self, lost_ids, unmatched_dets, score_fn, project_fn):
        used_dets = set()

        # Actualitzar candidatures existents
        for pid in list(self._candidates):
            if pid not in lost_ids:
                del self._candidates[pid]
                continue
            cand = self._candidates[pid]
            best_i, best_dist = None, CANDIDATE_MAX_DIST
            for i, det in enumerate(unmatched_dets):
                if i in used_dets:
                    continue
                dist = math.sqrt((cand.cx - det["cx"])**2
                                 + (cand.base_y - det["base_y"])**2)
                if dist < best_dist:
                    best_dist, best_i = dist, i
            if best_i is not None:
                det   = unmatched_dets[best_i]
                pos_m = project_fn(det["cx"], det["base_y"])
                score = score_fn(pid, pos_m)
                if score >= MIN_SCORE_TO_TRACK:
                    cand.update(det, score, pos_m)
                    used_dets.add(best_i)
                else:
                    cand.mark_missing()
            else:
                cand.mark_missing()

        # Crear candidatures noves
        remaining = [(i, d) for i, d in enumerate(unmatched_dets)
                     if i not in used_dets]
        for pid in lost_ids:
            if pid in self._candidates:
                continue
            best_score, best_i, best_pos = 0.0, None, None
            for i, det in remaining:
                pos_m = project_fn(det["cx"], det["base_y"])
                score = score_fn(pid, pos_m)
                if score > best_score:
                    best_score, best_i, best_pos = score, i, pos_m
            if best_i is not None and best_score >= MIN_SCORE_TO_TRACK:
                self._candidates[pid] = CandidateTrack(
                    pid, unmatched_dets[best_i], best_score, best_pos)
                print(f"[BUFFER] Nova candidatura J{pid} "
                      f"(score={best_score:.1f}, necessita {CONFIRM_THRESHOLD})")

        # Recollir confirmades i eliminar caducades
        confirmed, to_del = [], []
        for pid, cand in self._candidates.items():
            if cand.confirmed:
                confirmed.append((pid, cand.det))
                to_del.append(pid)
                print(f"[BUFFER] J{pid} CONFIRMAT "
                      f"(acc={cand.score_acc:.1f}, frames={cand.frames_seen})")
            elif cand.expired:
                to_del.append(pid)
                print(f"[BUFFER] J{pid} candidatura descartada")
        for pid in to_del:
            del self._candidates[pid]

        return confirmed

    def cancel(self, player_id):
        self._candidates.pop(player_id, None)

    @property
    def pending(self):
        return dict(self._candidates)


# ==========================================
# 7. TRACKER
# ==========================================

def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (aw*ah) + (bw*bh) - inter
    return inter / union if union > 0 else 0


class Tracker:
    def __init__(self, max_players=MAX_PLAYERS, homography=None):
        self.tracks      = []
        self.max_players = max_players
        self.homo        = homography
        self.all_ids     = set(range(1, max_players + 1))
        self._last_pos_m = {pid: None for pid in self.all_ids}
        self._buffer     = AssignmentBuffer()

    def set_homography(self, homography):
        self.homo = homography

    def _active_ids(self):
        return {t["id"] for t in self.tracks}

    def _lost_ids(self):
        return self.all_ids - self._active_ids()

    def _project(self, cx, base_y):
        if self.homo is None:
            return None
        return self.homo.project_point(cx, base_y)

    def update(self, mask):
        kernel_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
        mask_clean   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vert)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean   = cv2.erode(mask_clean, kernel_erode, iterations=1)

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                if (h / w if w > 0 else 0) < 0.8:
                    continue
                detections.append({
                    "bbox":   (x, y, w, h),
                    "cx":     x + w // 2,
                    "base_y": y + h,
                })

        detections = sorted(
            detections, key=lambda d: d["bbox"][2]*d["bbox"][3], reverse=True
        )[:self.max_players]

        matched_det_idx = set()

        # Pas 1: aparellar pistes actives
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

        # Pas 2: eliminar pistes massa perdudes
        self.tracks = [t for t in self.tracks if t["lost"] <= MAX_LOST_FRAMES]

        # Pas 3: buffer de confiança per a IDs perduts
        unmatched = [d for i, d in enumerate(detections)
                     if i not in matched_det_idx]
        lost_ids  = sorted(self._lost_ids())

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

    def get_ghost_tracks(self):
        """Pistes perdudes (posició home) + candidatures pendents."""
        active_ids = {t["id"] for t in self.tracks if t["lost"] == 0}
        ghosts = []

        for pid in self.all_ids:
            if pid in active_ids:
                continue
            if pid in self._buffer.pending:
                cand = self._buffer.pending[pid]
                ghosts.append({
                    "id": pid, "cx": cand.cx, "base_y": cand.base_y,
                    "pos_m": cand.pos_m, "ghost": "candidate", "pct": cand.pct,
                })
            else:
                last_m = self._last_pos_m.get(pid)
                pos_m  = last_m if last_m is not None else PLAYER_HOME.get(pid)
                if pos_m is None:
                    continue
                px = self.homo.project_point_inv(*pos_m) if self.homo else None
                ghosts.append({
                    "id": pid,
                    "cx":     int(px[0]) if px is not None else 0,
                    "base_y": int(px[1]) if px is not None else 0,
                    "pos_m": pos_m, "ghost": True, "pct": 0,
                })
        return ghosts


# ==========================================
# 8. HEATMAP
# ==========================================

class PlayerHeatmap:
    def __init__(self, resolution=80):
        self.res     = resolution
        self.w_px    = 10 * resolution
        self.h_px    = 20 * resolution
        self.RADI_PX = int(resolution * 0.4)
        self.maps    = {}

    def _ensure(self, pid):
        if pid not in self.maps:
            self.maps[pid] = np.zeros((self.h_px, self.w_px), dtype=np.float32)

    def update(self, pid, pos_m):
        if pos_m is None:
            return
        self._ensure(pid)
        x_m, y_m = pos_m
        if -1 <= x_m <= 11 and -1 <= y_m <= 21:
            px, py = int(x_m * self.res), int(y_m * self.res)
            if 0 <= px < self.w_px and 0 <= py < self.h_px:
                cv2.circle(self.maps[pid], (px, py), self.RADI_PX, 1.0, -1)

    def draw_court(self, img):
        r = self.res
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w-1, h-1), (255,255,255), 3)
        cv2.line(img, (0, h//2), (w, h//2), (255,255,255), 4)
        for y_m in [10 - 6.95, 10 + 6.95]:
            cv2.line(img, (0, int(y_m*r)), (w, int(y_m*r)), (255,255,255), 2)
        cv2.line(img, (w//2, int((10-6.95)*r)),
                 (w//2, int((10+6.95)*r)), (255,255,255), 2)

    def _process(self, raw):
        blurred = cv2.GaussianBlur(raw, (41, 41), 0)
        p95 = np.percentile(blurred, 95) or blurred.max()
        if p95 == 0:
            return None
        return (np.clip(blurred, 0, p95) / p95 * 255).astype(np.uint8)

    def save_individual(self, base="heatmap"):
        for pid, raw in self.maps.items():
            norm = self._process(raw)
            if norm is None:
                continue
            colored = cv2.applyColorMap(norm, get_player_colormap(pid))
            self.draw_court(colored)
            cv2.imwrite(f"{base}_jugador_{pid}.png", colored)
            print(f"[HEATMAP] Jugador {pid} guardat")

    def save_combined(self, path="heatmap_combinat.png"):
        if not self.maps:
            return
        combined = np.zeros((self.h_px, self.w_px, 3), dtype=np.float32)
        weight   = np.zeros((self.h_px, self.w_px), dtype=np.float32)
        for pid, raw in self.maps.items():
            norm = self._process(raw)
            if norm is None:
                continue
            intensity = norm.astype(np.float32) / 255.0
            for c, v in enumerate(get_player_color(pid)):
                combined[:,:,c] += intensity * v
            weight += intensity
        mask = weight > 0
        for c in range(3):
            combined[:,:,c][mask] = np.clip(
                combined[:,:,c][mask] / weight[mask] * 255, 0, 255)
        result = combined.astype(np.uint8)
        legend = np.zeros((50, self.w_px, 3), dtype=np.uint8)
        for pid in sorted(self.maps.keys()):
            x = (pid-1)*100
            cv2.rectangle(legend, (x+5,10), (x+25,35), get_player_color(pid), -1)
            cv2.putText(legend, f"J{pid}", (x+30,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_player_color(pid), 2)
        final = np.vstack([result, legend])
        self.draw_court(final)
        cv2.imwrite(path, final)
        print(f"[HEATMAP] Combinat guardat: {path}")


# ==========================================
# 9. MAIN
# ==========================================

if __name__ == "__main__":
    video_path    = "Data-Set/padel-data-labels/2022_BCN_FinalM_Retallat_1.mp4"
    MEDIAN_BG_PATH = "background_median.png"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error obrint el vídeo: {video_path}")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("El vídeo està buit.")
        exit()

    homo    = Homography()
    tracker = Tracker(max_players=MAX_PLAYERS)
    heatmap = PlayerHeatmap(resolution=80)

    static_filter   = StaticZoneFilter(warmup_frames=90)
    roi_filter      = RoiFilter(frame.shape, predefined_polys=ROI_EXCLUDE)
    temporal_filter = TemporalConsistencyFilter(window=5, min_hits=3)

    backSub = DualRateMOG2(
        history_play=400, var_threshold=60,
        lr_play=0.003, lr_pause=0.0, motion_thresh=600,
    )

    pause_detector = PauseDetector(window=25, play_thresh=600, pause_thresh=250)
    current_frame  = frame

    def on_point_start():
        print("[INICI PUNT] Reset suau MOG2...")
        for _ in range(50):
            backSub.mog.apply(current_frame, learningRate=0.05)

    pause_detector.on_play_start = on_point_start

    # Fons per mediana
    median_bg = MedianBackground(n_frames=150, threshold=18, skip=2)
    median_bg.load(MEDIAN_BG_PATH)

    if not median_bg.ready:
        print("[MEDIAN BG] Calculant fons (warmup)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while not median_bg.ready:
            ret, f = cap.read()
            if not ret:
                break
            median_bg.feed(f)
        median_bg.save(MEDIAN_BG_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Homografia
    ret, frame = cap.read()
    H = homo.compute_from_frame(frame)
    if H is None:
        exit()
    tracker.set_homography(homo)

    print("\n[INFO] Processant... Prem 'q' per aturar.")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame

        fg_mog2   = backSub.apply(frame, motion_pixels=pause_detector.motion_pixels)
        fg_median = median_bg.apply(frame)
        fg_mask   = cv2.bitwise_or(fg_mog2, fg_median)

        pause_detector.update(fg_mask)

        fg_mask = roi_filter.apply(fg_mask)
        fg_mask = static_filter.update(fg_mask)
        fg_mask = temporal_filter.update(fg_mask)
        fg_mask = edge_gate(fg_mask, frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

        active_tracks = tracker.update(fg_mask)
        ghost_tracks  = tracker.get_ghost_tracks()

        # Dibuixar tracks reals
        for t in active_tracks:
            pid   = t["id"]
            color = get_player_color(pid)
            x, y, w, h = t["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"J{pid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-lh-10), (x+lw+8, y), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label, (x+4, y-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.circle(frame, (t["cx"], t["base_y"]), 5, color, -1)
            pos_m = homo.project_point(t["cx"], t["base_y"])
            heatmap.update(pid, pos_m)

        # Dibuixar ghosts i candidatures
        for g in ghost_tracks:
            if g["cx"] == 0 and g["base_y"] == 0:
                continue
            pid   = g["id"]
            color = get_player_color(pid)
            cx, cy = g["cx"], g["base_y"]

            if g["ghost"] == "candidate":
                # Cercle puntejat + barra de progrés
                for angle in range(0, 360, 30):
                    a1, a2 = math.radians(angle), math.radians(angle + 15)
                    cv2.line(frame,
                             (int(cx + 16*math.cos(a1)), int(cy + 16*math.sin(a1))),
                             (int(cx + 16*math.cos(a2)), int(cy + 16*math.sin(a2))),
                             color, 1)
                bar_x = cx - 20
                cv2.rectangle(frame, (bar_x, cy+20), (bar_x+40, cy+25), (60,60,60), -1)
                cv2.rectangle(frame, (bar_x, cy+20),
                              (bar_x + int(40 * g["pct"] / 100), cy+25), color, -1)
                cv2.putText(frame, f"J{pid}? {g['pct']}%",
                            (cx+18, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.circle(frame, (cx, cy), 14, color, 1)
                cv2.putText(frame, f"J{pid}?", (cx+16, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            heatmap.update(pid, g["pos_m"])

        # Estat MOG2
        estat = "JUGANT" if backSub.is_playing else "PAUSA"
        color_estat = (0, 200, 100) if backSub.is_playing else (0, 120, 255)
        cv2.putText(frame, estat, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estat, 2)

        # Debug píxels
        cv2.putText(frame, f"px:{int(pause_detector.motion_pixels)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("Tracking Padel", frame)
        cv2.imshow("Mascara Jugadors", fg_mask)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    heatmap.save_individual("heatmap_padel")
    heatmap.save_combined("heatmap_padel_combinat.png")