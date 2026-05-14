import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict


# ─── Càrrega ──────────────────────────────────────────────────────────────────

def load_report(path, label):
    if not os.path.exists(path):
        print(f"[WARN] No s'ha trobat el report de {label}: {path}")
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame": int(row["frame"]),
                "gt":    int(row["gt"]),
                "det":   int(row["det"]),
                "tp":    int(row["tp"]),
                "fp":    int(row["fp"]),
                "fn":    int(row["fn"]),
            })
    rows.sort(key=lambda r: r["frame"])
    return rows


# ─── Mètriques ────────────────────────────────────────────────────────────────

def calc_metrics(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mota  = 1 - (fn + fp) / (tp + fn) if (tp + fn) > 0 else 0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1, "MOTA": mota}


def per_frame_metrics(rows):
    frames, precs, recs, f1s = [], [], [], []
    for r in rows:
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        re = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * re / (p + re) if (p + re) > 0 else 0
        frames.append(r["frame"])
        precs.append(p)
        recs.append(re)
        f1s.append(f)
    return frames, precs, recs, f1s


# ─── Print consola ────────────────────────────────────────────────────────────

def print_comparison(m1, m2):
    def arr(v2, v1):
        d = v2 - v1
        sym = "↑" if d > 0 else ("↓" if d < 0 else "→")
        return f"{sym}{abs(d)*100:.1f}pp"

    print("\n" + "=" * 65)
    print(f"  {'MÈTRICA':<15} {'PART 1 (MOG2)':<22} {'PART 2 (YOLO)':<22}")
    print("-" * 65)
    for k in ["TP", "FP", "FN"]:
        print(f"  {k:<15} {m1[k]:<22} {m2[k]:<22}")
    print("-" * 65)
    for k in ["Precision", "Recall", "F1", "MOTA"]:
        v1, v2 = m1[k], m2[k]
        print(f"  {k:<15} {v1*100:.2f}%{'':<15} {v2*100:.2f}%  {arr(v2,v1)}")
    print("=" * 65 + "\n")


# ─── Gràfiques ────────────────────────────────────────────────────────────────

COLORS = {
    "part1": "#4A90D9",
    "part2": "#E8613C",
    "bg":    "#0F1117",
    "card":  "#1A1D27",
    "grid":  "#2A2D3A",
    "text":  "#E8EAF0",
    "sub":   "#8B8FA8",
}


def setup_fig_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["card"],
        "axes.edgecolor":    COLORS["grid"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["sub"],
        "ytick.color":       COLORS["sub"],
        "grid.color":        COLORS["grid"],
        "text.color":        COLORS["text"],
        "legend.facecolor":  COLORS["card"],
        "legend.edgecolor":  COLORS["grid"],
        "font.family":       "monospace",
    })


def plot_bar_metrics(m1, m2, ax):
    keys   = ["Precision", "Recall", "F1", "MOTA"]
    v1     = [m1[k] * 100 for k in keys]
    v2     = [m2[k] * 100 for k in keys]
    x      = np.arange(len(keys))
    width  = 0.35

    b1 = ax.bar(x - width/2, v1, width, label="Part 1 · MOG2",
                color=COLORS["part1"], alpha=0.85, zorder=3)
    b2 = ax.bar(x + width/2, v2, width, label="Part 2 · YOLO",
                color=COLORS["part2"], alpha=0.85, zorder=3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8, color=COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Valor (%)", fontsize=9)
    ax.set_title("Mètriques globals", fontsize=11, fontweight="bold",
                 color=COLORS["text"], pad=10)
    ax.grid(axis="y", linewidth=0.5, zorder=0)
    ax.legend(fontsize=9)


def plot_tp_fp_fn(m1, m2, ax):
    cats   = ["TP", "FP", "FN"]
    v1     = [m1[k] for k in cats]
    v2     = [m2[k] for k in cats]
    x      = np.arange(len(cats))
    width  = 0.35

    ax.bar(x - width/2, v1, width, label="Part 1 · MOG2",
           color=COLORS["part1"], alpha=0.85, zorder=3)
    ax.bar(x + width/2, v2, width, label="Part 2 · YOLO",
           color=COLORS["part2"], alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_title("TP / FP / FN totals", fontsize=11, fontweight="bold",
                 color=COLORS["text"], pad=10)
    ax.set_ylabel("Nombre de deteccions", fontsize=9)
    ax.grid(axis="y", linewidth=0.5, zorder=0)
    ax.legend(fontsize=9)


def smooth(values, w=15):
    if len(values) < w:
        return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="same")


def plot_frame_metric(frames1, vals1, frames2, vals2, ax, title, ylabel):
    ax.plot(frames1, smooth(vals1), color=COLORS["part1"],
            linewidth=1.5, label="Part 1 · MOG2", alpha=0.9)
    ax.plot(frames2, smooth(vals2), color=COLORS["part2"],
            linewidth=1.5, label="Part 2 · YOLO", alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=COLORS["text"], pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(linewidth=0.4, zorder=0)
    ax.legend(fontsize=8)


def plot_radar(m1, m2, ax):
    labels = ["Precision", "Recall", "F1", "MOTA"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    v1 = [m1[k] for k in labels] + [m1[labels[0]]]
    v2 = [m2[k] for k in labels] + [m2[labels[0]]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7,
                       color=COLORS["sub"])

    ax.plot(angles, v1, color=COLORS["part1"], linewidth=2, label="Part 1 · MOG2")
    ax.fill(angles, v1, color=COLORS["part1"], alpha=0.15)
    ax.plot(angles, v2, color=COLORS["part2"], linewidth=2, label="Part 2 · YOLO")
    ax.fill(angles, v2, color=COLORS["part2"], alpha=0.15)

    ax.set_facecolor(COLORS["card"])
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title("Radar de mètriques", fontsize=11, fontweight="bold",
                 color=COLORS["text"], pad=20)


def plot_delta_f1(frames1, f1_1, frames2, f1_2, ax):
    """Diferència de F1 frame a frame (interpolada si els frames no coincideixen)."""
    set1 = dict(zip(frames1, f1_1))
    set2 = dict(zip(frames2, f1_2))
    common = sorted(set(frames1) & set(frames2))
    if not common:
        ax.text(0.5, 0.5, "No hi ha frames comuns", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["sub"])
        return
    delta = [set2[f] - set1[f] for f in common]
    delta_s = smooth(delta, 10)
    ax.axhline(0, color=COLORS["sub"], linewidth=0.8, linestyle="--")
    ax.fill_between(common, delta_s, 0,
                    where=[d >= 0 for d in delta_s],
                    color=COLORS["part2"], alpha=0.5, label="YOLO millor")
    ax.fill_between(common, delta_s, 0,
                    where=[d < 0 for d in delta_s],
                    color=COLORS["part1"], alpha=0.5, label="MOG2 millor")
    ax.set_title("Δ F1 per frame (YOLO − MOG2)", fontsize=11, fontweight="bold",
                 color=COLORS["text"], pad=8)
    ax.set_ylabel("Δ F1", fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    ax.grid(linewidth=0.4)
    ax.legend(fontsize=8)


def generate_plots(m1, m2, rows1, rows2, out_path):
    setup_fig_style()

    f1s1, f1s2 = per_frame_metrics(rows1), per_frame_metrics(rows2)
    frames1, precs1, recs1, f1_1 = f1s1
    frames2, precs2, recs2, f1_2 = f1s2

    fig = plt.figure(figsize=(18, 14), facecolor=COLORS["bg"])
    fig.suptitle("Comparació Part 1 (MOG2)  vs  Part 2 (YOLOv8)",
                 fontsize=16, fontweight="bold", color=COLORS["text"], y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # Fila 0
    ax0 = fig.add_subplot(gs[0, 0])
    plot_bar_metrics(m1, m2, ax0)

    ax1 = fig.add_subplot(gs[0, 1])
    plot_tp_fp_fn(m1, m2, ax1)

    ax_radar = fig.add_subplot(gs[0, 2], polar=True)
    plot_radar(m1, m2, ax_radar)

    # Fila 1
    ax2 = fig.add_subplot(gs[1, :2])
    plot_frame_metric(frames1, precs1, frames2, precs2, ax2,
                      "Precisió per frame", "Precisió")

    ax3 = fig.add_subplot(gs[1, 2])
    plot_frame_metric(frames1, recs1, frames2, recs2, ax3,
                      "Recall per frame", "Recall")

    # Fila 2
    ax4 = fig.add_subplot(gs[2, :2])
    plot_delta_f1(frames1, f1_1, frames2, f1_2, ax4)

    ax5 = fig.add_subplot(gs[2, 2])
    plot_frame_metric(frames1, f1_1, frames2, f1_2, ax5,
                      "F1 per frame", "F1")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    print(f"[INFO] Gràfica guardada: {out_path}")
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara els reports de validació de Part 1 i Part 2."
    )
    parser.add_argument("--part1",
                        default="Back-End/Part_1/validation_report.csv",
                        help="Path al CSV de Part 1 (MOG2)")
    parser.add_argument("--part2",
                        default="Back-End/Part2/validation_report_yolo.csv",
                        help="Path al CSV de Part 2 (YOLO)")
    parser.add_argument("--out",   default="comparison.png",
                        help="Fitxer de sortida per la gràfica")
    args = parser.parse_args()

    rows1 = load_report(args.part1, "Part 1 (MOG2)")
    rows2 = load_report(args.part2, "Part 2 (YOLO)")

    if rows1 is None and rows2 is None:
        print("[ERROR] Cap dels dos reports existeix. Res a comparar.")
        sys.exit(1)

    if rows1 is None:
        print("[INFO] Només hi ha report de Part 2. Mostrant mètriques individuals.")
        m2 = calc_metrics(rows2)
        print("\n── RESULTATS PART 2 (YOLOv8) ──")
        for k, v in m2.items():
            val = f"{v*100:.2f}%" if isinstance(v, float) else str(v)
            print(f"  {k:<12}: {val}")
        sys.exit(0)

    if rows2 is None:
        print("[INFO] Només hi ha report de Part 1. Mostrant mètriques individuals.")
        m1 = calc_metrics(rows1)
        print("\n── RESULTATS PART 1 (MOG2) ──")
        for k, v in m1.items():
            val = f"{v*100:.2f}%" if isinstance(v, float) else str(v)
            print(f"  {k:<12}: {val}")
        sys.exit(0)

    m1 = calc_metrics(rows1)
    m2 = calc_metrics(rows2)

    print_comparison(m1, m2)
    generate_plots(m1, m2, rows1, rows2, args.out)


if __name__ == "__main__":
    main()