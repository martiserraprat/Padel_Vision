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

ROI_EXCLUDE = [
    # Exemple: franja superior on hi ha el vidre del fons
    # np.array([[0, 0], [1920, 0], [1920, 120], [0, 120]], dtype=np.int32),
    # Exemple: franja inferior
    # np.array([[0, 960], [1920, 960], [1920, 1080], [0, 1080]], dtype=np.int32),
]

def get_player_color(player_id):
    return PLAYER_COLORS[(player_id - 1) % len(PLAYER_COLORS)]

def get_player_colormap(player_id):
    return HEATMAP_COLORMAPS[(player_id - 1) % len(HEATMAP_COLORMAPS)]

# ==========================================
# 2. CLASSE HOMOGRAFIA
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
# 3A. FILTRE DE ZONES ESTÀTIQUES (warmup)
# ==========================================
class StaticZoneFilter:
    """Aprèn automàticament quines zones sempre fan soroll i les elimina."""
    def __init__(self, warmup_frames=90):
        self.warmup_frames = warmup_frames
        self.noise_mask = None
        self.frame_count = 0
        self.accumulator = None

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
# 3B. FILTRE ROI (exclou zones de cristall)
# ==========================================
class RoiFilter:
    """
    Exclou polígons fixos de la màscara (cristalls, xarxa, publicitat...).
    Es configura una sola vegada des del primer frame amb una UI interactiva,
    o bé es pot passar una llista de polígons predefinits a ROI_EXCLUDE.
    """
    def __init__(self, frame_shape, predefined_polys=None):
        h, w = frame_shape[:2]
        self.mask = np.ones((h, w), dtype=np.uint8) * 255  # tot permès per defecte

        if predefined_polys:
            for poly in predefined_polys:
                cv2.fillPoly(self.mask, [poly], 0)
            print(f"[ROI] {len(predefined_polys)} zones excloses carregades.")

    def setup_interactive(self, frame):
        """
        Permet dibuixar polígons d'exclusió clicant sobre el frame.
        - Clic esquerre: afegir punt al polígon actual
        - 'c': tancar i confirmar el polígon actual
        - 'r': descartar el polígon actual
        - ENTER: finalitzar i guardar tots els polígons
        """
        print("\n[ROI] Dibuixa les zones a EXCLOURE (cristalls, xarxa...)")
        print("  Clic esquerre → afegir punt")
        print("  'c'           → tancar polígon actual")
        print("  'r'           → descartar polígon actual")
        print("  ENTER         → finalitzar")

        h, w = frame.shape[:2]
        self.mask = np.ones((h, w), dtype=np.uint8) * 255

        current_poly = []
        all_polys    = []
        clone        = frame.copy()

        cv2.namedWindow("Defineix zones ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Defineix zones ROI", 1280, 720)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                current_poly.append((x, y))

        cv2.setMouseCallback("Defineix zones ROI", on_click)

        while True:
            display = clone.copy()

            # Dibuixar polígons ja confirmats (vermell semitransparent)
            overlay = display.copy()
            for poly in all_polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
            for poly in all_polys:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 0, 255), 2)

            # Dibuixar polígon en construcció (groc)
            for i, p in enumerate(current_poly):
                cv2.circle(display, p, 4, (0, 220, 255), -1)
                if i > 0:
                    cv2.line(display, current_poly[i-1], p, (0, 220, 255), 1)

            # Text d'ajuda
            cv2.putText(display, f"Poligs confirmats: {len(all_polys)}  |  Punts actuals: {len(current_poly)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "c=tancar  r=descartar  ENTER=finalitzar",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Defineix zones ROI", display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('c') and len(current_poly) >= 3:
                all_polys.append(list(current_poly))
                current_poly.clear()
                print(f"[ROI] Polígon {len(all_polys)} confirmat.")

            elif key == ord('r'):
                current_poly.clear()
                print("[ROI] Polígon descartat.")

            elif key in [13, 10]:  # ENTER
                if current_poly and len(current_poly) >= 3:
                    all_polys.append(list(current_poly))
                break

        cv2.destroyWindow("Defineix zones ROI")

        # Aplicar tots els polígons a la màscara
        for poly in all_polys:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], 0)

        print(f"[ROI] {len(all_polys)} zones d'exclusió aplicades.\n")

    def apply(self, fg_mask):
        return cv2.bitwise_and(fg_mask, self.mask)

# ==========================================
# 3C. FILTRE DE CONSISTÈNCIA TEMPORAL
# ==========================================
class TemporalConsistencyFilter:
    """
    Manté un buffer dels últims N frames i només deixa passar els píxels
    que apareixen com a foreground en almenys min_hits d'ells.
    Elimina parpelleig de cristalls i reflexos intermitents.
    """
    def __init__(self, window=5, min_hits=3):
        self.window   = window
        self.min_hits = min_hits
        self.buffer   = deque(maxlen=window)

    def update(self, fg_mask):
        self.buffer.append(fg_mask.copy())

        if len(self.buffer) < self.window:
            return fg_mask  # encara escalfant, passa sense filtrar

        stack = np.stack(list(self.buffer), axis=0).astype(np.uint16)
        count = (stack > 0).sum(axis=0).astype(np.uint8)

        _, consistent = cv2.threshold(
            count, self.min_hits - 1, 255, cv2.THRESH_BINARY
        )
        return consistent.astype(np.uint8)

# ==========================================
# 3D. PORTA DE CONTORNS (edge gate)
# ==========================================
def edge_gate(fg_mask, frame, low=30, high=90, dilate_px=15):
    """
    Filtra la màscara deixant només les deteccions que coincideixen
    amb contorns forts al frame original.
    Els cristalls generen canvis difusos; els jugadors, contorns durs.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_px, dilate_px)
    )
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    return cv2.bitwise_and(fg_mask, edges_dilated)

# ==========================================
# 4. CLASSE TRACKER AMB IDs RECICLATS
# ==========================================
def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0

class Tracker:
    def __init__(self, max_players=MAX_PLAYERS):
        self.tracks      = []
        self.max_players = max_players
        self.available_ids = deque(range(1, max_players + 1))
        self.recycled_ids  = deque()

    def _claim_id(self):
        if self.recycled_ids:
            return self.recycled_ids.popleft()
        if self.available_ids:
            return self.available_ids.popleft()
        return None

    def _release_id(self, pid):
        self.recycled_ids.appendleft(pid)

    def update(self, mask):
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
        mask_clean  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vert)

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
                    "base_y": y + h
                })

        detections = sorted(
            detections, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True
        )
        detections = detections[:self.max_players]

        matched_det_ids = set()

        for t in self.tracks:
            best_det = None
            best_iou = IOU_THRESHOLD
            min_dist = 70

            for i, det in enumerate(detections):
                if i in matched_det_ids:
                    continue

                iou = compute_iou(t["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = i

                if best_det is None:
                    dist = math.sqrt(
                        (t["cx"] - det["cx"])**2 + (t["base_y"] - det["base_y"])**2
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
                matched_det_ids.add(best_det)
            else:
                t["lost"] += 1

        tracks_vius = []
        for t in self.tracks:
            if t["lost"] <= MAX_LOST_FRAMES:
                tracks_vius.append(t)
            else:
                self._release_id(t["id"])
                print(f"[TRACKER] ID {t['id']} alliberat (perdut)")
        self.tracks = tracks_vius

        active_ids = {t["id"] for t in self.tracks}
        for i, det in enumerate(detections):
            if i not in matched_det_ids:
                if len(active_ids) >= self.max_players:
                    continue
                new_id = self._claim_id()
                if new_id is None:
                    continue
                self.tracks.append({
                    "id":     new_id,
                    "bbox":   det["bbox"],
                    "cx":     det["cx"],
                    "base_y": det["base_y"],
                    "lost":   0
                })
                active_ids.add(new_id)
                print(f"[TRACKER] ID {new_id} assignat")

        return [t for t in self.tracks if t["lost"] == 0]

# ==========================================
# 5. CLASSE HEATMAP PER JUGADOR
# ==========================================
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

# ==========================================
# 6. EXECUCIÓ PRINCIPAL (MAIN)
# ==========================================
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

    # --- Inicialitzar objectes ---
    homo      = Homography()
    tracker   = Tracker(max_players=MAX_PLAYERS)
    heatmap   = PlayerHeatmap(resolution=80)

    # Filtre 1: zones estàtiques apreses automàticament
    static_filter = StaticZoneFilter(warmup_frames=90)

    # Filtre 2: ROI manual de cristalls
    # Opció A — polígons predefinits a ROI_EXCLUDE (dalt del fitxer)
    # Opció B — dibuix interactiu en iniciar (descomenta la línia setup_interactive)
    roi_filter = RoiFilter(frame.shape, predefined_polys=ROI_EXCLUDE)
    roi_filter.setup_interactive(frame)   # ← descomenta per dibuixar manualment

    # Filtre 3: consistència temporal (window=5, cal aparèixer a ≥3 frames)
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

    print("\n[INFO] Processant el vídeo... Prem 'q' per aturar-ho.")

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Substracció de fons KNN
        fg_mask = backSub.apply(frame)

        # 2. Eliminar ombres (valor 127 → 0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 3. Excloure zones fixes de cristall (ROI)
        fg_mask = roi_filter.apply(fg_mask)

        # 4. Eliminar soroll estàtic après durant el warmup
        fg_mask = static_filter.update(fg_mask)

        # 5. Consistència temporal: descartar parpelleig de cristalls
        fg_mask = temporal_filter.update(fg_mask)

        # 6. Porta de contorns: descartar reflexos difusos
        fg_mask = edge_gate(fg_mask, frame)

        # 7. Neteja morfològica final
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

        # 8. Tracker
        active_tracks = tracker.update(fg_mask)

        # 9. Dibuixar resultats
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