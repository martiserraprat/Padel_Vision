"""
validate_tracker.py
===================
Valida el tracker de pàdel comparant les deteccions del sistema (main.py)
amb les anotacions ground-truth del CSV.

Mètriques calculades per frame:
  - True Positives  (TP): detecció prou propera a una GT bbox
  - False Positives (FP): deteccions sense GT corresponent
  - False Negatives (FN): jugadors GT no detectats
  - Precision, Recall, F1
  - MOTA (Multiple Object Tracking Accuracy)
  - IoU mig entre parelles TP

Ús:
    python validate_tracker.py \
        --video  Data-Set/padel-data-labels/2022_BCN_FinalM_Retallat_1.mp4 \
        --csv    annotations.csv \
        --iou    0.30 \
        --show              # opcional: mostra finestra en temps real
"""

import argparse
import csv
import math
import cv2
import numpy as np
from collections import defaultdict, deque

try:
    from main import (
        Homography, Tracker, StaticZoneFilter, RoiFilter,
        TemporalConsistencyFilter, edge_gate, ROI_EXCLUDE,
        MAX_PLAYERS, get_player_color
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("[WARN] No s'ha pogut importar main.py. "
          "El validador funcionarà en mode STANDALONE (només mètriques CSV).")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    """boxA, boxB = (x, y, w, h)"""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def load_gt_from_csv(csv_path: str) -> dict:
    """
    Llegeix el CSV i agrupa les anotacions per (video, frame_id).
    Retorna:
        gt[video][frame_name] = [ (x,y,w,h), ... ]
    """
    gt = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video  = row["video"].strip()
            frame  = row["frame"].strip()          # p.ex. "frame_000450.PNG"
            x      = float(row["bbox_x"])
            y      = float(row["bbox_y"])
            w      = float(row["bbox_w"])
            h      = float(row["bbox_h"])
            gt[video][frame].append((x, y, w, h))
    return gt


def match_detections(gt_boxes, det_boxes, iou_thresh=0.30):
    """
    Matching greedy (major IoU primer) entre GT i deteccions.
    Retorna: (tp_list, fp_count, fn_count, iou_list)
        tp_list  : llista d'IoU de cada parella TP
        fp_count : nombre de deteccions sense GT
        fn_count : nombre de GT sense detecció
    """
    matched_gt  = set()
    matched_det = set()
    iou_list    = []

    # Construïm matriu d'IoU
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for di, db in enumerate(det_boxes):
            iou = compute_iou(gb, db)
            if iou >= iou_thresh:
                pairs.append((iou, gi, di))

    pairs.sort(reverse=True)

    for iou, gi, di in pairs:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        iou_list.append(iou)

    tp = len(iou_list)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes)  - tp
    return tp, fp, fn, iou_list


def draw_validation(frame_img, gt_boxes, det_boxes, tp_pairs):
    """
    Dibuixa sobre el frame:
      - GT       → verd
      - Detectat → blau (TP) / vermell (FP)
    """
    vis = frame_img.copy()
    for gb in gt_boxes:
        x, y, w, h = [int(v) for v in gb]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 220, 0), 2)
        cv2.putText(vis, "GT", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)

    matched_det_idx = set(tp_pairs)
    for di, db in enumerate(det_boxes):
        x, y, w, h = [int(v) for v in db]
        color = (255, 100, 0) if di in matched_det_idx else (0, 0, 255)
        label = "TP" if di in matched_det_idx else "FP"
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, label, (x, y+h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


# ─── Validació STANDALONE (sense vídeo) ──────────────────────────────────────

def validate_standalone(csv_path: str, iou_thresh: float, video_filter: str = None):
    """
    Valida únicament llegint el CSV: simula deteccions perfectes i mesura
    la consistència interna (útil per comprovar que el CSV és correcte).
    En producció, substitueix `det_boxes` per les deteccions reals del tracker.
    """
    print("\n[STANDALONE] Validació amb bboxes GT com a deteccions (sanity-check)")
    gt = load_gt_from_csv(csv_path)

    total_tp = total_fp = total_fn = 0
    all_iou  = []
    id_switches = 0

    for video, frames in gt.items():
        if video_filter and video_filter not in video:
            continue
        for frame_name, gt_boxes in sorted(frames.items()):
            # En mode standalone: les deteccions = GT mateixos → sempre TP perfecte
            det_boxes = list(gt_boxes)
            tp, fp, fn, ious = match_detections(gt_boxes, det_boxes, iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_iou.extend(ious)

    _print_summary(total_tp, total_fp, total_fn, all_iou, id_switches)


# ─── Validació COMPLETA amb el pipeline real ──────────────────────────────────

def validate_with_pipeline(video_path: str, csv_path: str,
                            iou_thresh: float, show: bool,
                            video_filter: str = None):
    """
    Processa el vídeo amb el pipeline complet i compara frame a frame
    amb les anotacions del CSV.
    """
    gt_all = load_gt_from_csv(csv_path)

    # Determina quin "video" del CSV correspon a aquest fitxer
    # (cerca per nom base sense extensió)
    import os
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Busca la clau al CSV que contingui el base_name
    gt_video = None
    for key in gt_all.keys():
        if key in base_name or base_name in key:
            gt_video = key
            break
    if gt_video is None:
        # Prova amb la primera entrada disponible
        gt_video = next(iter(gt_all))
        print(f"[WARN] No s'ha trobat '{base_name}' al CSV. "
              f"Usant '{gt_video}' com a referència.")
    else:
        print(f"[INFO] Usant ground-truth de '{gt_video}'")

    gt_frames = gt_all[gt_video]

    # ── Detectar offset: primer frame del CSV ──────────────────────────────
    # Els noms de frame al CSV son "frame_XXXXXX.PNG"
    # Extraiem el número mínim per saber des d'on comença el GT
    frame_numbers_in_csv = []
    for fname in gt_frames.keys():
        try:
            num = int(''.join(filter(str.isdigit, fname.split('.')[0])))
            frame_numbers_in_csv.append(num)
        except ValueError:
            pass

    if frame_numbers_in_csv:
        first_gt_frame = min(frame_numbers_in_csv)
        last_gt_frame  = max(frame_numbers_in_csv)
        # El vídeo està retallat: frame 1 del vídeo = first_gt_frame del CSV
        # Per tant: nom_csv = frame_video + (first_gt_frame - 1)
        frame_offset = first_gt_frame - 1  # ex: 449 si CSV comença a frame_000450
        print(f"[INFO] CSV cobreix frames {first_gt_frame} → {last_gt_frame}")
        print(f"[INFO] Offset aplicat: frame_video + {frame_offset} = frame_csv")
    else:
        first_gt_frame = 1
        last_gt_frame  = 999999
        frame_offset   = 0
        print("[WARN] No s'ha pogut detectar l'offset. Usant offset 0.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No es pot obrir: {video_path}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] No s'ha pogut llegir el frame inicial.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rebobina al principi del vídeo

    # ── Inicialitzar pipeline ──
    homo           = Homography()
    tracker        = Tracker(max_players=MAX_PLAYERS)
    static_filter  = StaticZoneFilter(warmup_frames=90)
    roi_filter     = RoiFilter(first_frame.shape, predefined_polys=ROI_EXCLUDE)
    roi_filter.setup_interactive(first_frame)
    temporal_filter= TemporalConsistencyFilter(window=5, min_hits=3)
    backSub        = cv2.createBackgroundSubtractorMOG2(
                         history=1700, varThreshold=100, detectShadows=True)
    kernel_open    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    H = homo.compute_from_frame(first_frame)
    if H is None:
        print("[ERROR] Homografia cancel·lada.")
        cap.release()
        return

    # ── Estadístiques ──
    total_tp = total_fp = total_fn = 0
    all_iou  = []
    id_switches = 0
    frame_results = []

    # frame_idx compta els frames del vídeo (comença a 1)
    frame_idx = 0

    print(f"\n[INFO] Processant '{video_path}'... Prem 'q' per aturar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Nom de frame al CSV: frame_video (1,2,3...) + offset = frame_csv
        # Ex: vídeo retallat a s15 → frame 1 del vídeo = frame_000450 del CSV
        frame_csv  = frame_idx + frame_offset
        frame_name = f"frame_{frame_csv:06d}.PNG"
        gt_boxes   = gt_frames.get(frame_name, [])

        # ── Pipeline de detecció ──
        fg = backSub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = roi_filter.apply(fg)
        fg = static_filter.update(fg)
        fg = temporal_filter.update(fg)
        fg = edge_gate(fg, frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
        active_tracks = tracker.update(fg)

        det_boxes = [t["bbox"] for t in active_tracks]

        # ── Matching GT ↔ Deteccions ──
        tp, fp, fn, ious = match_detections(gt_boxes, det_boxes, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_iou.extend(ious)
        frame_results.append({
            "frame": frame_csv, "gt": len(gt_boxes),
            "det": len(det_boxes), "tp": tp, "fp": fp, "fn": fn
        })

        # Atura quan acabin els frames del CSV
        if frame_csv >= last_gt_frame:
            print(f"[INFO] Finalitzat: últim frame GT ({last_gt_frame}) processat.")
            break

        # ── Visualització opcional ──
        if show:
            tp_det_idx = set(range(tp))
            vis = draw_validation(frame, gt_boxes, det_boxes, tp_det_idx)
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
            cv2.putText(vis, f"Frame vídeo:{frame_idx} CSV:{frame_csv} | GT:{len(gt_boxes)} Det:{len(det_boxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(vis, f"TP:{tp} FP:{fp} FN:{fn} | Prec:{prec:.2f} Rec:{rec:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            cv2.imshow("Validació Tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    _print_summary(total_tp, total_fp, total_fn, all_iou, id_switches)
    _save_frame_report(frame_results)


# ─── Resum final ──────────────────────────────────────────────────────────────

def _print_summary(tp, fp, fn, iou_list, id_switches):
    precision = tp/(tp+fp)      if (tp+fp) > 0 else 0
    recall    = tp/(tp+fn)      if (tp+fn) > 0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    mean_iou  = sum(iou_list)/len(iou_list) if iou_list else 0
    # MOTA = 1 - (FN + FP + IDSW) / GT_total
    gt_total  = tp + fn
    mota      = 1 - (fn + fp + id_switches) / gt_total if gt_total > 0 else 0

    print("\n" + "="*55)
    print("       RESUM DE VALIDACIÓ DEL TRACKER")
    print("="*55)
    print(f"  True  Positives  (TP) : {tp}")
    print(f"  False Positives  (FP) : {fp}")
    print(f"  False Negatives  (FN) : {fn}")
    print(f"  ID Switches      (IS) : {id_switches}")
    print(f"  Precision              : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall                 : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1-Score               : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  IoU mig (TP)           : {mean_iou:.4f}")
    print(f"  MOTA                   : {mota:.4f}  ({mota*100:.1f}%)")
    print("="*55 + "\n")


def _save_frame_report(frame_results, path="validation_report.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","gt","det","tp","fp","fn"])
        writer.writeheader()
        writer.writerows(frame_results)
    print(f"[INFO] Informe per frame guardat a: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valida el tracker de pàdel contra anotacions CSV")
    parser.add_argument("--video",  default=None,
        help="Ruta al vídeo (si no s'indica, mode standalone)")
    parser.add_argument("--csv",    required=True,
        help="Ruta al CSV d'anotacions ground-truth")
    parser.add_argument("--iou",    type=float, default=0.30,
        help="Llindar IoU per considerar un TP (default: 0.30)")
    parser.add_argument("--show",   action="store_true",
        help="Mostra finestra de visualització en temps real")
    parser.add_argument("--video-filter", default=None,
        help="Filtra pel nom de vídeo al CSV (standalone)")
    args = parser.parse_args()

    if args.video and PIPELINE_AVAILABLE:
        validate_with_pipeline(
            video_path=args.video,
            csv_path=args.csv,
            iou_thresh=args.iou,
            show=args.show,
            video_filter=args.video_filter
        )
    else:
        if args.video and not PIPELINE_AVAILABLE:
            print("[WARN] Pipeline no disponible; executant mode standalone.")
        validate_standalone(
            csv_path=args.csv,
            iou_thresh=args.iou,
            video_filter=args.video_filter
        )


if __name__ == "__main__":
    main()