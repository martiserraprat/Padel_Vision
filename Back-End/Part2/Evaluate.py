"""
Evaluate.py - Part 2 (YOLOv8)
Execució:
python Evaluate.py --video ..\\..\\Data-Set\\padel-data-labels\\2022_BCN_FinalM_Retallat_1.mp4 --csv ..\\..\\Data-Set\\padel-data-labels\\GT_Retallat_15_120.csv --show
"""

import argparse
import csv
import os
import sys
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Primer Part2 (on és TrackerYOLO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as main2
TrackerYOLO      = main2.TrackerYOLO
build_roi_mask   = main2.build_roi_mask
detection_in_roi = main2.detection_in_roi

# Després Part1 (on és Homography, Tracker, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Part_1'))
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

# ── Configuració YOLO ─────────────────────────────────────────────────────────
YOLO_MODEL     = "yolov8n.pt"
CONF_THRESHOLD = 0.4
IOU_NMS        = 0.5

PART1_REPORT_CSV = "../Part_1/validation_report.csv"

def load_part1_results(path):
    if not os.path.exists(path):
        print(f"[WARN] No s'ha trobat el report de Part 1: {path}")
        return None
    tp = fp = fn = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tp += int(row["tp"])
            fp += int(row["fp"])
            fn += int(row["fn"])
    return {"TP": tp, "FP": fp, "FN": fn}

PART1_RESULTS = load_part1_results(PART1_REPORT_CSV)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_gt(csv_path):
    gt = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = row["frame"].strip()
            gt[frame].append((
                float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]),
            ))
    return gt


def match(gt_boxes, det_boxes, iou_thresh):
    pairs = sorted(
        [(compute_iou(g, d), gi, di)
         for gi, g in enumerate(gt_boxes)
         for di, d in enumerate(det_boxes)
         if compute_iou(g, d) >= iou_thresh],
        reverse=True
    )
    matched_g, matched_d, ious = set(), set(), []
    for iou, gi, di in pairs:
        if gi in matched_g or di in matched_d:
            continue
        matched_g.add(gi); matched_d.add(di); ious.append(iou)
    tp = len(ious)
    return tp, len(det_boxes)-tp, len(gt_boxes)-tp, ious


def draw_frame(frame, gt_boxes, det_boxes, tp_count):
    vis = frame.copy()
    for g in gt_boxes:
        x, y, w, h = map(int, g)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,220,0), 2)
        cv2.putText(vis, "GT", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
    for i, d in enumerate(det_boxes):
        x, y, w, h = map(int, d)
        color = (255,100,0) if i < tp_count else (0,0,255)
        label = "TP" if i < tp_count else "FP"
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
        cv2.putText(vis, label, (x, y+h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def calc_metrics(tp, fp, fn, iou_list):
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    miou = sum(iou_list)/len(iou_list) if iou_list else 0
    mota = 1 - (fn+fp)/(tp+fn) if (tp+fn) > 0 else 0
    return prec, rec, f1, miou, mota


def print_summary(tp, fp, fn, iou_list):
    prec, rec, f1, miou, mota = calc_metrics(tp, fp, fn, iou_list)
    print("\n" + "="*50)
    print(f"  TP        : {tp}")
    print(f"  FP        : {fp}")
    print(f"  FN        : {fn}")
    print(f"  Precision : {prec:.3f} ({prec*100:.1f}%)")
    print(f"  Recall    : {rec:.3f} ({rec*100:.1f}%)")
    print(f"  F1        : {f1:.3f} ({f1*100:.1f}%)")
    print(f"  IoU mig   : {miou:.3f}")
    print(f"  MOTA      : {mota:.3f} ({mota*100:.1f}%)")
    print("="*50 + "\n")


def print_comparison(tp2, fp2, fn2, iou_list2):
    if PART1_RESULTS is None:
        print("\n[INFO] No hi ha resultats de Part 1 per comparar.\n")
        return
    
    tp1, fp1, fn1 = PART1_RESULTS["TP"], PART1_RESULTS["FP"], PART1_RESULTS["FN"]
    p1, r1, f1_1, _, mota1 = calc_metrics(tp1, fp1, fn1, [])
    p2, r2, f1_2, miou2, mota2 = calc_metrics(tp2, fp2, fn2, iou_list2)

    def diff(v2, v1):
        d = v2 - v1
        return f"({'↑' if d > 0 else '↓'}{abs(d)*100:.1f}%)"

    print("\n" + "="*60)
    print(f"  {'MÈTRICA':<15} {'PART 1 (MOG2)':<20} {'PART 2 (YOLO)':<20}")
    print("-"*60)
    print(f"  {'TP':<15} {tp1:<20} {tp2:<20}")
    print(f"  {'FP':<15} {fp1:<20} {fp2:<20}")
    print(f"  {'FN':<15} {fn1:<20} {fn2:<20}")
    print(f"  {'Precision':<15} {p1*100:.1f}%{'':<14} {p2*100:.1f}% {diff(p2,p1)}")
    print(f"  {'Recall':<15} {r1*100:.1f}%{'':<14} {r2*100:.1f}% {diff(r2,r1)}")
    print(f"  {'F1':<15} {f1_1*100:.1f}%{'':<14} {f1_2*100:.1f}% {diff(f1_2,f1_1)}")
    print(f"  {'MOTA':<15} {mota1*100:.1f}%{'':<14} {mota2*100:.1f}% {diff(mota2,mota1)}")
    print("="*60 + "\n")


def save_report(rows, path="validation_report_yolo.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame","gt","det","tp","fp","fn"])
        w.writeheader(); w.writerows(rows)
    print(f"[INFO] Informe guardat: {path}")


# ─── Validació principal ──────────────────────────────────────────────────────

def validate(video_path, csv_path, iou_thresh, show):
    gt_frames = load_gt(csv_path)

    nums = []
    for fname in gt_frames:
        try:
            nums.append(int(''.join(filter(str.isdigit, fname.split('.')[0]))))
        except ValueError:
            pass
    first_gt = min(nums) if nums else 1
    last_gt  = max(nums) if nums else 999999
    offset   = first_gt - 1
    print(f"[INFO] GT: frames {first_gt} → {last_gt} (offset={offset})")

    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No es pot obrir: {video_path}"); return

    ret, first_frame = cap.read()
    if not ret: return

    roi_mask = build_roi_mask(first_frame.shape)
    homo     = Homography()
    tracker  = TrackerYOLO(max_players=MAX_PLAYERS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    H = homo.compute_from_frame(first_frame)
    if H is None:
        cap.release(); return
    tracker.set_homography(homo)

    total_tp = total_fp = total_fn = 0
    all_iou, rows = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        frame_csv  = frame_idx + offset
        frame_name = f"frame_{frame_csv:06d}.PNG"
        gt_boxes   = gt_frames.get(frame_name, [])

        results = model(
            frame, classes=[0],
            conf=CONF_THRESHOLD, iou=IOU_NMS, verbose=False,
        )[0]

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
        det_boxes     = [t["bbox"] for t in active_tracks]

        tp, fp, fn, ious = match(gt_boxes, det_boxes, iou_thresh)
        total_tp += tp; total_fp += fp; total_fn += fn
        all_iou.extend(ious)
        rows.append({"frame": frame_csv, "gt": len(gt_boxes),
                     "det": len(det_boxes), "tp": tp, "fp": fp, "fn": fn})

        if show:
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
            vis  = draw_frame(frame, gt_boxes, det_boxes, tp)
            cv2.putText(vis, f"frame {frame_csv} | GT:{len(gt_boxes)} Det:{len(det_boxes)} TP:{tp} FP:{fp} FN:{fn}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(vis, f"Prec:{prec:.2f} Rec:{rec:.2f}",
                        (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
            cv2.imshow("Validació YOLOv8", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_csv >= last_gt: break

    cap.release()
    cv2.destroyAllWindows()

    print("\n── RESULTATS PART 2 (YOLOv8) ──")
    print_summary(total_tp, total_fp, total_fn, all_iou)

    print("── COMPARACIÓ PART 1 vs PART 2 ──")
    print_comparison(total_tp, total_fp, total_fn, all_iou)

    save_report(rows)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv",   required=True)
    parser.add_argument("--iou",   type=float, default=0.30)
    parser.add_argument("--show",  action="store_true")
    args = parser.parse_args()
    validate(args.video, args.csv, args.iou, args.show)