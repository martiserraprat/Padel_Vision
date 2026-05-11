"""
Evaluate.py
python Evaluate.py --video ../../Data-Set/padel-data-labels/2022_BCN_FinalM_Retallat_1.mp4 --csv ../../Data-Set/padel-data-labels/GT_Retallat_15_120.csv --show
"""

import argparse
import csv
import cv2
import numpy as np
from collections import defaultdict

from main import (
    Homography, Tracker, StaticZoneFilter, RoiFilter,
    TemporalConsistencyFilter, edge_gate, ROI_EXCLUDE, MAX_PLAYERS,
    DualRateMOG2, PauseDetector, MedianBackground,
)

MEDIAN_BG_PATH = "background_median.png"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def load_gt(csv_path):
    """Retorna {frame_name: [(x,y,w,h), ...]}"""
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
    """Matching greedy. Retorna (tp, fp, fn, [ious])."""
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


def print_summary(tp, fp, fn, iou_list):
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    miou = sum(iou_list)/len(iou_list) if iou_list else 0
    mota = 1 - (fn+fp)/(tp+fn) if (tp+fn) > 0 else 0
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


def save_report(rows, path="validation_report.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame","gt","det","tp","fp","fn"])
        w.writeheader(); w.writerows(rows)
    print(f"[INFO] Informe guardat: {path}")


# ─── Validació principal ──────────────────────────────────────────────────────

def validate(video_path, csv_path, iou_thresh, show):
    import os
    gt_frames = load_gt(csv_path)

    # Detectar offset: el CSV pot começar a frame_000450, etc.
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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No es pot obrir: {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        return

    # ── Pipeline (igual que main.py) ──
    homo            = Homography()
    tracker         = Tracker(max_players=MAX_PLAYERS)
    static_filter   = StaticZoneFilter(warmup_frames=90)
    roi_filter      = RoiFilter(first_frame.shape, predefined_polys=ROI_EXCLUDE)
    temporal_filter = TemporalConsistencyFilter(window=5, min_hits=3)
    backSub         = DualRateMOG2(history_play=400, var_threshold=60,
                                   lr_play=0.003, lr_pause=0.0, motion_thresh=600)
    pause_detector  = PauseDetector(window=25, play_thresh=600, pause_thresh=250)
    kernel_open     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    median_bg = MedianBackground(n_frames=150, threshold=18, skip=2)
    median_bg.load(MEDIAN_BG_PATH)
    if not median_bg.ready:
        print("[MEDIAN BG] Calculant fons...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while not median_bg.ready:
            ret, f = cap.read()
            if not ret: break
            median_bg.feed(f)
        median_bg.save(MEDIAN_BG_PATH)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    H = homo.compute_from_frame(first_frame)
    if H is None:
        cap.release(); return
    tracker.set_homography(homo)

    current_frame = first_frame
    def on_point_start():
        for _ in range(50):
            backSub.mog.apply(current_frame, learningRate=0.05)
    pause_detector.on_play_start = on_point_start

    # ── Bucle ──
    total_tp = total_fp = total_fn = 0
    all_iou, rows = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx  += 1
        current_frame = frame

        frame_csv  = frame_idx + offset
        frame_name = f"frame_{frame_csv:06d}.PNG"
        gt_boxes   = gt_frames.get(frame_name, [])

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
            cv2.imshow("Validació", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_csv >= last_gt: break

    cap.release()
    cv2.destroyAllWindows()
    print_summary(total_tp, total_fp, total_fn, all_iou)
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